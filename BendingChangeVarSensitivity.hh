#ifndef XHATSENSITIVITY_HH
#define XHATSENSITIVITY_HH

#include "InflatablePeriodicUnit.hh"
#include <cmath>
#include <rotation_optimization.hh>
// #include <MeshFEM/ElasticSolidRotExtrap.hh>
#include "MeshFEM_temp_RotExtrap.hh"


struct x_hat_sensitivity {
#if INFLATABLES_LONG_DOUBLE
    using Real = long double;
#else
    using Real = double;
#endif
    using  V3d = typename InflatablePeriodicUnit::V3d;
    using  V2d = typename InflatablePeriodicUnit::V2d;
    using  VXd = typename InflatablePeriodicUnit::VXd;
    using MX2d = typename InflatablePeriodicUnit::MX2d;
    using MX3d = typename InflatablePeriodicUnit::MX3d;
    using  M3d = typename InflatablePeriodicUnit::M3d;

    using RO = rotation_optimization<Real>;

    static constexpr size_t JacobianRows = 3;
    static constexpr size_t JacobianCols = 3 + 1 + 1;
    // Derivative of x_hat w.r.t. x, kappa, alpha.
    Eigen::Matrix<Real, JacobianRows, JacobianCols> jacobian;

    // hessian[i] holds the Hessian of the i^th component of x_hat^T
    // with respect to (x, kappa, alpha).
    std::array<Eigen::Matrix<Real, JacobianCols, JacobianCols>, JacobianRows> hessian;

    const size_t kappa_offset = 3, alpha_offset = 4;

    /////////////////////

    V3d get_d_omega_d_alpha(Real alpha, Real kappa, V3d u) {
        if (std::abs(kappa) < 1e-14) {
            return V3d::Zero();
        }
        V3d d_omega_d_alpha;
        // d_omega_d_alpha << m_kappa * (-sin(m_alpha)) * u(0) + m_kappa * cos(m_alpha) * u(1), 
        //                    m_kappa * (-sin(m_alpha)) * u(1),
        //                    m_kappa * (-sin(m_alpha)) * u(2);
        d_omega_d_alpha << u(0) * (sin(alpha) * sin(alpha) - cos(alpha) * cos(alpha)) - 2 * cos(alpha) * sin(alpha) * u(1),
                            -2 * cos(alpha) * sin(alpha) * u(0) + u(1) * (- sin(alpha) * sin(alpha) + cos(alpha) * cos(alpha)), 
                            0.0;
        d_omega_d_alpha *= kappa;
        return d_omega_d_alpha;
    }

    V3d get_d_omega_d_alpha_d_kappa(Real alpha, Real kappa, V3d u) {
        V3d d_omega_d_alpha_d_kappa;
        // d_omega_d_alpha << m_kappa * (-sin(m_alpha)) * u(0) + m_kappa * cos(m_alpha) * u(1), 
        //                    m_kappa * (-sin(m_alpha)) * u(1),
        //                    m_kappa * (-sin(m_alpha)) * u(2);
        d_omega_d_alpha_d_kappa << u(0) * (sin(alpha) * sin(alpha) - cos(alpha) * cos(alpha)) - 2 * cos(alpha) * sin(alpha) * u(1),
                            -2 * cos(alpha) * sin(alpha) * u(0) + u(1) * (- sin(alpha) * sin(alpha) + cos(alpha) * cos(alpha)), 
                            0.0;
        return d_omega_d_alpha_d_kappa;
    }

    V3d get_d_omega_d_alpha_d_alpha(Real alpha, Real kappa, V3d u) {
        if (std::abs(kappa) < 1e-14) {
            return V3d::Zero();
        }
        V3d d_omega_d_alpha_d_alpha;
        d_omega_d_alpha_d_alpha << 2 * u(0) * sin(2 * alpha) - 2 * u(1) * cos(2 * alpha),
                          -2 * u(0) * cos(2 * alpha) - 2 * u(1) * sin(2 * alpha),
                            0.0;

        d_omega_d_alpha_d_alpha *= kappa;
        return d_omega_d_alpha_d_alpha;
    }

    M3d get_d_omega_d_alpha_d_x(Real alpha, Real kappa) {
        if (std::abs(kappa) < 1e-14) {
            return M3d::Zero();
        }
        M3d d_omega_d_alpha_d_x;
        d_omega_d_alpha_d_x << sin(alpha) * sin(alpha) - cos(alpha) * cos(alpha), -2 * cos(alpha) * sin(alpha), 0.0,
                            -2 * cos(alpha) * sin(alpha), cos(alpha) * cos(alpha) - sin(alpha) * sin(alpha), 0.0,
                               0.0, 0.0, 0.0;
        d_omega_d_alpha_d_x *= kappa;
        return d_omega_d_alpha_d_x;
    }

    template<typename F>
    M3d tensor_helper(F &&d_f1_d_f2_v_d_f3, const V3d &delta) {
        M3d result;
        result.col(0) = delta.transpose() * d_f1_d_f2_v_d_f3(V3d::UnitX());
        result.col(1) = delta.transpose() * d_f1_d_f2_v_d_f3(V3d::UnitY());
        result.col(2) = delta.transpose() * d_f1_d_f2_v_d_f3(V3d::UnitZ());
        return result;
    }

    void update(const Real alpha, const Real kappa, const M3d grad_omega, const M3d axis_mat, V3d u, bool evalHessian) {

        jacobian.setZero();

        Real h = u(2);
        u(2) = 0.0;

        V3d z = V3d::UnitZ();

        M3d flatten = M3d::Identity() - V3d::UnitZ() * V3d::UnitZ().transpose();

        V3d omega = grad_omega * u;
        Real theta_sq = omega.dot(omega);
        Real theta = sqrt(theta_sq);

        auto get_d_omega_cross_omega_cross_v_d_omega = [&](const V3d v) -> M3d {
            return (omega.transpose() * v * M3d::Identity() + omega * v.transpose() - 2 * v * omega.transpose()).transpose();
        };
        
        Real f1 = two_cos_minus_2_plus_theta_sin_div_theta_pow_4(theta, theta_sq);
        Real f2 = one_minus_cos_div_theta_sq(theta, theta_sq);
        Real f3 = three_sin_minus_theta_times_two_plus_cos_div_theta_pow_5(theta, theta_sq);
        Real f4 = theta_minus_sin_div_theta_cubed(theta, theta_sq);

        // d_x_hat_d_omega

        M3d d1 = omega * (omega.cross(u)).transpose();
        M3d d2 = RO::cross_product_matrix(u);
        M3d d3 = omega * (omega.cross(omega.cross(u))).transpose();
        M3d d4 = get_d_omega_cross_omega_cross_v_d_omega(u);
        // M3d d_x_hat_d_omega = f1 * d1 + f2 * d2 + f3 * d3 + f4 * d4;
        
        // M3d d_x_hat_d_x = M3d::Identity();
        // d_x_hat_d_x += (one_minus_cos_div_theta_sq(theta, theta_sq) * RO::cross_product_matrix(omega) + theta_minus_sin_div_theta_cubed(theta, theta_sq) * (RO::cross_product_matrix(omega) * RO::cross_product_matrix(omega))).transpose();

        Real g1 = eight_plus_theta_sq_minus_eight_cos_minus_five_theta_sin_div_theta_pow_6(theta, theta_sq);
        Real g3 = theta_sq_minus_15_sin_plus_8_theta_plus_7_theta_cos_div_theta_pow_7(theta, theta_sq);


        M3d d_x_hat_d_omega = f1 * d1 + f2 * d2 + f3 * d3 + f4 * d4;

        M3d d_omega_d_x = grad_omega.transpose();
        M3d d_omega_d_x_d_kappa = axis_mat.transpose();


        M3d d_x_hat_d_x = M3d::Identity();
        d_x_hat_d_x += f2 * RO::cross_product_matrix(omega).transpose();

        d_x_hat_d_x += f4 * (RO::cross_product_matrix(omega) * RO::cross_product_matrix(omega)).transpose();

        d_x_hat_d_x = flatten * d_x_hat_d_x;

        M3d d_x_hat_d_omega_times_d_omega_d_x2d_times_d_x2d_d_x = d_omega_d_x * d_x_hat_d_omega;

        ///////////////////////////
        // Normal terms
        M3d d_Re3_d_omega = RO::grad_rotated_vector(omega, z).transpose();

        M3d d_n_d_omega = h * d_Re3_d_omega;

        V3d d_n_d_x3 = RO::rotated_vector(omega, z);
        M3d d_n_d_x = d_omega_d_x * d_n_d_omega;


        ///////////////////////////

        ////////////////////////////////////

        V3d d_omega_d_kappa = axis_mat * u;
        jacobian.template block<3, 1>(0, kappa_offset) += d_x_hat_d_omega.transpose() * d_omega_d_kappa;

        V3d d_omega_d_alpha = get_d_omega_d_alpha(alpha, kappa, u);
        jacobian.template block<3, 1>(0, alpha_offset) += d_x_hat_d_omega.transpose() * d_omega_d_alpha;
        
        jacobian.template block<3, 3>(0, 0) += d_x_hat_d_x.transpose() + d_x_hat_d_omega_times_d_omega_d_x2d_times_d_x2d_d_x.transpose();

        // // Normal terms.
        jacobian.template block<3, 1>(0, kappa_offset) += d_n_d_omega.transpose() * d_omega_d_kappa;

        jacobian.template block<3, 1>(0, alpha_offset) += d_n_d_omega.transpose() * d_omega_d_alpha;

        jacobian.template block<3, 3>(0, 0) += d_n_d_x.transpose();

        jacobian.template block<3, 1>(0, 2) += d_n_d_x3.transpose();

        ////////////////////////////////////
        if (!evalHessian) return;

        auto d_d_omega_d_omega_multiply_v = [&](const V3d v) -> M3d {
            M3d d4_d_omega_multiply_v = u * v.transpose() + v * u.transpose() - 2 * u.transpose() * v * M3d::Identity();
            if (theta_sq < 1e-6) {
                return f4 * d4_d_omega_multiply_v;
            }
            M3d result;
            result.setZero();
            M3d A = RO::cross_product_matrix(u).transpose();
            // Cosine term.
            result += f1 * omega * (A.transpose() * v).transpose();
            result += g1 * omega * (omega * (omega.cross(u)).transpose() * v).transpose();
            result += f1 * (omega.transpose() * A.transpose() * v * M3d::Identity() + omega * (v.transpose() * A)).transpose();

            // Sine term.
            result += f4 * d4_d_omega_multiply_v;
            result += f3 * omega * (d4 * v).transpose();

            result += g3  * omega  * (omega * (omega.cross(omega.cross(u))).transpose() * v).transpose();

            result += f3 * ((omega.transpose() * u * omega - omega.transpose() * omega * u).transpose() * v * M3d::Identity() + omega.transpose() * v * omega * u.transpose() + omega.transpose() * u * omega * v.transpose() - 2 * u.transpose() * v * omega * omega.transpose()).transpose();

            return result;
        };

        auto d_Rv_d_omega = [&](const V3d v) -> M3d {
            if (theta_sq < 1e-6) {
                return f2 * RO::cross_product_matrix(v).transpose() * flatten;
            }
            M3d result = f1 * omega * (v.cross(omega)).transpose() + f2 * RO::cross_product_matrix(v).transpose();
            // Sine term.
            M3d A = RO::cross_product_matrix(omega) * RO::cross_product_matrix(omega);

            result += f3 * omega * (v.transpose() * A);
            // result += f4 * (omega * v.transpose() + omega.transpose() * v * M3d::Identity() - 2 * v * omega.transpose());
            result += f4 * get_d_omega_cross_omega_cross_v_d_omega(v);

            return result * flatten;
        };
        
        auto d_x_hat_d_omega_v_d_x = [&](const V3d v) -> M3d {
            if (theta_sq < 1e-6) {
                return flatten * f2 * RO::cross_product_matrix(v);
            }
            M3d A = RO::cross_product_matrix(omega);
            M3d B = RO::cross_product_matrix(v);
            M3d result = f2 * B + f1 * (omega * v.transpose() * A).transpose();

            // Sine term.
            M3d double_omega_cross = A * A;
            result += f4 * get_d_omega_cross_omega_cross_v_d_omega(v).transpose();
            result += f3 * (omega * (v.transpose() * double_omega_cross)).transpose();
            return flatten * result;
        };

        ///////////////////

        V3d d_x_hat_d_alpha_d_alpha = d_omega_d_alpha.transpose() * tensor_helper(d_d_omega_d_omega_multiply_v, d_omega_d_alpha) + get_d_omega_d_alpha_d_alpha(alpha, kappa, u).transpose() * d_x_hat_d_omega;
        V3d d_x_hat_d_kappa_d_kappa = d_omega_d_kappa.transpose() * tensor_helper(d_d_omega_d_omega_multiply_v, d_omega_d_kappa);

        V3d d_omega_d_alpha_d_kappa = get_d_omega_d_alpha_d_kappa(alpha, kappa, u);
        V3d d_x_hat_d_alpha_d_kappa = d_omega_d_alpha.transpose() * tensor_helper(d_d_omega_d_omega_multiply_v, d_omega_d_kappa) + d_omega_d_alpha_d_kappa.transpose() * d_x_hat_d_omega; 


        // Terms related to x 


        M3d d_omega_d_alpha_d_x = get_d_omega_d_alpha_d_x(alpha, kappa);

        M3d d_x_hat_d_alpha_d_x = d_omega_d_x * tensor_helper(d_d_omega_d_omega_multiply_v, d_omega_d_alpha) + d_omega_d_alpha_d_x * d_x_hat_d_omega;
        d_x_hat_d_alpha_d_x += tensor_helper(d_Rv_d_omega, d_omega_d_alpha);

        M3d d_x_hat_d_kappa_d_x = d_omega_d_x * tensor_helper(d_d_omega_d_omega_multiply_v, d_omega_d_kappa);
        d_x_hat_d_kappa_d_x += d_omega_d_x_d_kappa * d_x_hat_d_omega;
        d_x_hat_d_kappa_d_x += tensor_helper(d_Rv_d_omega, d_omega_d_kappa);

        
        // d_x_hat_d_x_d_x_0, d_x_hat_d_x_d_x_1, d_x_hat_d_x_d_x_2
        std::vector<M3d> d_x_hat_d_x_d_x_iterate_x = std::vector<M3d>({M3d::Zero(), M3d::Zero(), M3d::Zero()});

        for (size_t i = 0; i < 2; i++) {
            d_x_hat_d_x_d_x_iterate_x[i] += d_omega_d_x * tensor_helper(d_d_omega_d_omega_multiply_v, d_omega_d_x.row(i));
        }
        for (size_t i = 0; i < 3; i++) {
            d_x_hat_d_x_d_x_iterate_x[i] += tensor_helper(d_Rv_d_omega, d_omega_d_x.row(i));
        }

        // d_x_hat_d_omega

        // d_x_hat_d_x_d_x_0 += d_omega_d_x * tensor_helper(d_x_hat_d_omega_v_d_x, V3d::UnitX());
        // d_x_hat_d_x_d_x_1 += d_omega_d_x * tensor_helper(d_x_hat_d_omega_v_d_x, V3d::UnitY());
        // d_x_hat_d_x_d_x_2 += d_omega_d_x * tensor_helper(d_x_hat_d_omega_v_d_x, V3d::UnitZ());


        // For d_x_hat_d_x the double term is not zero
        M3d d_x_hat_0_d_x_d_x = (d_x_hat_d_omega_v_d_x(V3d::UnitX()) * d_omega_d_x.transpose()).transpose();
        M3d d_x_hat_1_d_x_d_x = (d_x_hat_d_omega_v_d_x(V3d::UnitY()) * d_omega_d_x.transpose()).transpose();
        M3d d_x_hat_2_d_x_d_x = (d_x_hat_d_omega_v_d_x(V3d::UnitZ()) * d_omega_d_x.transpose()).transpose();
        std::vector<M3d> d_x_hat_d_x_d_x_iterate_x_hat = std::vector<M3d>({d_x_hat_0_d_x_d_x, d_x_hat_1_d_x_d_x, d_x_hat_2_d_x_d_x});


        ///////////////////////////
        // Normal terms

        auto d_Re3_d_omega_d_omega_multiply_v = [&](const V3d &v) -> M3d {
            return RO::d_contract_hess_rotated_vector(omega, z, v);
        };


        V3d d_n_d_alpha_d_alpha = h * d_omega_d_alpha.transpose() * tensor_helper(d_Re3_d_omega_d_omega_multiply_v, d_omega_d_alpha) + get_d_omega_d_alpha_d_alpha(alpha, kappa, u).transpose() * d_n_d_omega;
        V3d d_n_d_kappa_d_kappa = h * d_omega_d_kappa.transpose() * tensor_helper(d_Re3_d_omega_d_omega_multiply_v, d_omega_d_kappa);

        V3d d_n_d_alpha_d_kappa = h * d_omega_d_alpha.transpose() * tensor_helper(d_Re3_d_omega_d_omega_multiply_v, d_omega_d_kappa) + d_omega_d_alpha_d_kappa.transpose() * d_n_d_omega; 


            // Terms related to x 


      // Terms related to x 


        // M3d d_x_hat_d_kappa_d_x = d_omega_d_x * tensor_helper(d_d_omega_d_omega_multiply_v, d_omega_d_kappa);
        // M3d d_omega_d_kappa_d_x = d_omega_d_x / kappa;
        // d_x_hat_d_kappa_d_x += d_omega_d_kappa_d_x * d_x_hat_d_omega;
        // d_x_hat_d_kappa_d_x += tensor_helper(d_Rv_d_omega, d_omega_d_kappa);

        M3d d_n_d_alpha_d_x = h * (d_omega_d_x * tensor_helper(d_Re3_d_omega_d_omega_multiply_v, d_omega_d_alpha) + d_omega_d_alpha_d_x * d_Re3_d_omega);
        V3d d_n_d_alpha_d_x3 = d_omega_d_alpha.transpose() * d_Re3_d_omega;

        M3d d_n_d_kappa_d_x = h * (d_omega_d_x * tensor_helper(d_Re3_d_omega_d_omega_multiply_v, d_omega_d_kappa) + d_omega_d_x_d_kappa * d_Re3_d_omega);
        V3d d_n_d_kappa_d_x3 = d_omega_d_kappa.transpose() * d_Re3_d_omega;


        // d_n_d_x_d_x_0, d_n_d_x_d_x_1, d_n_d_x_d_x_2
        std::vector<M3d> d_n_d_x_d_x_iterate_x = std::vector<M3d>({M3d::Zero(), M3d::Zero(), M3d::Zero()});

        for (size_t i = 0; i < 2; i++) {
            d_n_d_x_d_x_iterate_x[i] += h * d_omega_d_x * tensor_helper(d_Re3_d_omega_d_omega_multiply_v, d_omega_d_x.row(i));
        }

            d_n_d_x_d_x_iterate_x[2] += d_omega_d_x * d_Re3_d_omega;

        auto d_n_d_omega_v_d_x = [&](const V3d v) -> M3d {
            M3d result = M3d::Zero();
            result.row(2) = d_omega_d_x * d_Re3_d_omega * v;
            return result;
        };

        M3d d_n_0_d_x_d_x = (d_n_d_omega_v_d_x(V3d::UnitX()) * d_omega_d_x.transpose()).transpose();
        M3d d_n_1_d_x_d_x = (d_n_d_omega_v_d_x(V3d::UnitY()) * d_omega_d_x.transpose()).transpose();
        M3d d_n_2_d_x_d_x = (d_n_d_omega_v_d_x(V3d::UnitZ()) * d_omega_d_x.transpose()).transpose();
        std::vector<M3d> d_n_d_x_d_x_iterate_n = std::vector<M3d>({d_n_0_d_x_d_x, d_n_1_d_x_d_x, d_n_2_d_x_d_x});


        ///////////////////////////
        for (size_t i = 0; i < 3; ++i) { hessian[i].setZero(); }

        size_t x3_offset = 2;

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                hessian[i].template block<3, 1>(0, j) += d_x_hat_d_x_d_x_iterate_x[j].col(i);
            }

            hessian[i].template block<3, 3>(0, 0) += d_x_hat_d_x_d_x_iterate_x_hat[i];

            hessian[i].template block<3, 1>(0, kappa_offset) = d_x_hat_d_kappa_d_x.col(i);
            hessian[i].template block<1, 3>(kappa_offset, 0) = d_x_hat_d_kappa_d_x.col(i);

            hessian[i].template block<3, 1>(0, alpha_offset) = d_x_hat_d_alpha_d_x.col(i);
            hessian[i].template block<1, 3>(alpha_offset, 0) = d_x_hat_d_alpha_d_x.col(i);

            hessian[i](kappa_offset, kappa_offset) = d_x_hat_d_kappa_d_kappa(i);
            hessian[i](alpha_offset, alpha_offset) = d_x_hat_d_alpha_d_alpha(i);

            hessian[i](kappa_offset, alpha_offset) = d_x_hat_d_alpha_d_kappa(i);
            hessian[i](alpha_offset, kappa_offset) = d_x_hat_d_alpha_d_kappa(i);



            // Normal terms

            for (size_t j = 0; j < 3; ++j) {
                hessian[i].template block<3, 1>(0, j) += d_n_d_x_d_x_iterate_x[j].col(i);
            }

            hessian[i].template block<3, 3>(0, 0) += d_n_d_x_d_x_iterate_n[i];
            

            hessian[i].template block<3, 1>(0, kappa_offset) += d_n_d_kappa_d_x.col(i);
            hessian[i].template block<1, 3>(kappa_offset, 0) += d_n_d_kappa_d_x.col(i);

            hessian[i](2, kappa_offset) += d_n_d_kappa_d_x3(i);
            hessian[i](kappa_offset, 2) += d_n_d_kappa_d_x3(i);

            hessian[i].template block<3, 1>(0, alpha_offset) += d_n_d_alpha_d_x.col(i);
            hessian[i].template block<1, 3>(alpha_offset, 0) += d_n_d_alpha_d_x.col(i);

            hessian[i](2, alpha_offset) += d_n_d_alpha_d_x3(i);
            hessian[i](alpha_offset, 2) += d_n_d_alpha_d_x3(i);

            hessian[i](kappa_offset, kappa_offset) += d_n_d_kappa_d_kappa(i);
            hessian[i](alpha_offset, alpha_offset) += d_n_d_alpha_d_alpha(i);

            hessian[i](kappa_offset, alpha_offset) += d_n_d_alpha_d_kappa(i);
            hessian[i](alpha_offset, kappa_offset) += d_n_d_alpha_d_kappa(i);
        }

    }
};

#endif