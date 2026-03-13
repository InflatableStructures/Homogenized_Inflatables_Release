////////////////////////////////////////////////////////////////////////////////
// ElasticSolidRotExtrap.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Applies a change of variables around a source configuration wherein the
//  skew symmetric "infinitesimal rotation part" of each element's average
//  velocity gradient is extrapolated as a finite rotation (while the symmetric
//  part is extrapolated linearly). The resulting per-element-node
//  displacements are averaged onto the nodes to obtain the continuous
//  displacement field.
//
//  The resulting nonlinear energy landscape has the same energy and gradient
//  as the original when the source configuration is up-to-date, but the
//  Hessian will differ. In particular, the nullspace corresponding to rigid
//  rotation (that is lost in stressed configurations under the conventional,
//  trivial parametrization) is restored; we expect this to particularly
//  benefit the solution of problems employing pin constraints to eliminate
//  rigid motion and more generally expect accelerated convergence due to
//  reduction of linearization artifacts in the finite steps made by the line
//  search.
//
//  Because of the local nature of the reparametrization, the Hessian *sparsity
//  pattern* is identical to the underlying `ElasticSolid`, meaning no
//  additional expense is incurred in the linear solve.
//  However, additional indefiniteness may be introduced.
//
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/27/2022 11:16:20
*///////////////////////////////////////////////////////////////////////////////
#ifndef MeshFEM_temp_RotExtrap_HH
#define MeshFEM_temp_RotExtrap_HH

#include <rotation_optimization.hh>

/////////////
// Temperorily avoid using files in EnergyDensity from MeshFEM
// Dimension-specific calculations
template<typename _Real, size_t N>
struct CRQuantities;

template<typename _Real>
struct CRQuantities<_Real, 3> {
    using GType    = Eigen::Matrix<_Real, 3, 3>;
    using IRotType = Eigen::Matrix<_Real, 3, 1>; // infinitesimal rotation representation
    using Mat      = Eigen::Matrix<_Real, 3, 3>;

    template<typename Derived>
    static GType getG(const Eigen::MatrixBase<Derived> &S) { return S.trace() * GType::Identity() - S; }

    template<typename Derived>
    static GType getGinv(const Eigen::MatrixBase<Derived> &G) { return G.inverse(); }

    // Extract a vector representing the skew symmetric part 0.5 (A - A^T)
    //     0 -c  b      [a]
    //     c  0 -a  ==> [b]
    //    -b  a  0      [c]
    // (This is the vector `w` whose cross product `w x v` equals `0.5 (A - A^T) v`.)
    template<typename Derived>
    static IRotType sk_inv(const Eigen::MatrixBase<Derived> &A) {
        return IRotType(0.5 * (A(2, 1) - A(1, 2)),
                        0.5 * (A(0, 2) - A(2, 0)),
                        0.5 * (A(1, 0) - A(0, 1)));
    }

    // B * sk(w)
    template<typename Derived>
    static Mat right_mul_sk(const Eigen::MatrixBase<Derived> &B, const IRotType &w) {
        return B.rowwise().cross(w);
    }
};

template<typename _Real>
struct CRQuantities<_Real, 2> {
    using GType    = _Real;
    using IRotType = _Real;
    using Mat      = Eigen::Matrix<_Real, 2, 2>;

    template<typename Derived>
    static GType getG(const Eigen::MatrixBase<Derived> &S) { return S.trace(); }

    static GType getGinv(_Real G) { return 1.0 / G; }

    // Extract a scalar representing the skew symmetric part 0.5 (A - A^T)
    //   0 -a
    //   a  0
    // (This scalar represents the counterclockwise infinitesimal rotation
    //  applied by `0.5 (A - A^T)`
    template<typename Derived>
    static IRotType sk_inv(const Eigen::MatrixBase<Derived> &A) {
        return 0.5 * (A(1, 0) - A(0, 1));
    }

    // B * sk(w)
    template<typename Derived>
    static Mat right_mul_sk(const Eigen::MatrixBase<Derived> &B, const IRotType &w) {
        Mat result;
        result << w * B.col(1),
                 -w * B.col(0);
        return result;
    }
};

/////////////

template<typename Real, size_t N>
struct RotExtrap;

template<typename Real>
struct RotExtrap<Real, 3> {
    using M3d = Eigen::Matrix<Real, 3, 3>;
    using V3d = Eigen::Matrix<Real, 3, 1>;
    using RO  = rotation_optimization<Real>;
    using WField = Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using WEntry = V3d;

    static WEntry get_w(const M3d &grad_u) { return CRQuantities<Real, 3>::sk_inv(grad_u); }

    static std::pair<M3d, V3d> extrapolate(const M3d &grad_u, const V3d &xbar, const V3d &ubar) {
        // Determine the rotation and stretching parts
        V3d w(get_w(grad_u));      // cross-product vector representation of the skew symmetric part
        V3d w_cross_c = w.cross(xbar) - ubar;
        Real thetaSq = w.squaredNorm();
        Real theta = std::sqrt(thetaSq);
        return std::make_pair(
                RO::rotation_matrix(w), // Extrapolate the rotation part
                V3d(-sinc(theta, thetaSq) * w_cross_c - one_minus_cos_div_theta_sq(theta, thetaSq) * w.cross(w_cross_c))); // Velocity of vector connecting centroid to center of rotation
    }

    // Calculate `Rtilde(w) u`
    static V3d apply_Rtilde(const WEntry &w, const V3d &u) {
        Real thetaSq = w.squaredNorm();
        Real theta = std::sqrt(thetaSq);
        V3d wxu = w.cross(u);
        return one_minus_cos_div_theta_sq(theta, thetaSq) * wxu + theta_minus_sin_div_theta_cubed(theta, thetaSq) * w.cross(wxu);
    }

    static V3d modal_warp_correction(const WEntry &w, const V3d &u) {
        return apply_Rtilde(w, u);
    }

    static M3d nodal_warp_derivative(const WEntry &w_k, const V3d &g_k, const V3d &u_k) {
        Real theta_sq = w_k.squaredNorm();
        Real theta = std::sqrt(theta_sq);

        V3d w_cross_u = w_k.cross(u_k);
        V3d w_cross_g = w_k.cross(g_k);

        M3d result = (0.5 * (two_cos_minus_2_plus_theta_sin_div_theta_pow_4(theta, theta_sq) * g_k.dot(w_cross_u)
                                - three_sin_minus_theta_times_two_plus_cos_div_theta_pow_5(theta, theta_sq) * w_cross_g.dot(w_cross_u))) * RO::cross_product_matrix(w_k);
        result += one_minus_cos_div_theta_sq(theta, theta_sq) * (g_k * u_k.transpose());
        result += theta_minus_sin_div_theta_cubed(theta, theta_sq) * (g_k * w_cross_u.transpose() - w_cross_g * u_k.transpose());
        return 0.5 * (result - result.transpose()); // actual result is the skew symmetric part
    }
};

template<typename Real>
struct RotExtrap<Real, 2> {
    using M2d =  Eigen::Matrix<Real, 2, 2>;
    using V2d =  Eigen::Matrix<Real, 2, 1>;
    using RO  =  rotation_optimization<Real>;
    using WField = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using WEntry = Eigen::Matrix<Real, 1, 1>;

    static WEntry get_w(const M2d &grad_u) { return WEntry(CRQuantities<Real, 2>::sk_inv(grad_u)); }

    static std::pair<M2d, V2d> extrapolate(const M2d &/* grad_u */, const V2d &/* xbar */, const V2d &/* ubar */) {
        // Determine the rotation and stretching parts

#if 0
        Real w = get_w(grad_u)[0];
        const Real theta_sq = w * w;
        const Real theta    = std::abs(w);
        return stretch * RO::cos(theta,  theta_sq)
            + w * (w.transpose() * stretch) * RO::one_minus_cos_div_theta_sq(theta, theta_sq) - stretch.colwise().cross(w * RO::sinc(theta, theta_sq));
#endif
        throw std::runtime_error("Unimplemented");
    }

    static V2d apply_Rtilde(const WEntry &/* w */, const V2d &/* u */) {
        throw std::runtime_error("Unimplemented");
    }

    static V2d modal_warp_correction(const WEntry &/* w */, const V2d &/* u */) {
        throw std::runtime_error("Unimplemented");
    }

    static M2d nodal_warp_derivative(const WEntry &/* w_k */, const V2d &/* g_k */, const V2d &/* u_k */) {
        throw std::runtime_error("Unimplemented");
    }
};

#endif /* end of include guard: MeshFEM_temp_RotExtrap_HH */
