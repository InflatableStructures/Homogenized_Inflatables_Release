#ifndef INFLATABLEMIDSURFACEPERIODICUNIT_HH
#define INFLATABLEMIDSURFACEPERIODICUNIT_HH
#include "InflatablePeriodicUnit.hh"

// Constrain the fluctuation displacement variables to have zero average heights through a linear change of variables. 
// The change of variable equation is as follows:
//   1  0  0 0 0 0
//  -1  1  0 0 0 0
//   0 -1  1 0 0 0 
//  ....

struct InflatableMidSurfacePeriodicUnit;

struct InflatableMidSurfacePeriodicUnit {
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
    using  M2d = typename InflatablePeriodicUnit::M2d;
    using TMatrix = TripletMatrix<Triplet<Real>>;

    using Mesh = typename InflatablePeriodicUnit::Mesh;
    using EnergyType = typename InflatablePeriodicUnit::EnergyType;
    using EnergyDensity = typename InflatablePeriodicUnit::EnergyDensity;
    static constexpr size_t NONE = std::numeric_limits<size_t>::max();

    static constexpr size_t N = 3;

    static constexpr bool average_z_reparametrization = true;

    InflatableMidSurfacePeriodicUnit(const std::shared_ptr<Mesh> &inMesh, const std::vector<bool> &fusedVtx = std::vector<bool>(), Real epsilon = 1e-7);

    void setMaterial(const EnergyDensity &psi) {
        ipu.setMaterial(psi);
    }

    // SuiteSparseMatrix midSurfaceToPeriodicPatchMapTranspose() const { return m_midSurfaceToPeriodicPatchMapTranspose; }
    // void m_constructMidSurfaceToPeriodicPatchMapTranspose();
    void m_constructMidSurfaceToPeriodicPatchMapTranspose_all_vars();

    // reduced to unreduced
    const VXd applyTransformation(const VXd &PPV) const {
        return m_midSurfaceToPeriodicPatchMapTranspose_all_vars.apply(PPV, /* transpose */ true);
    }
    // Unreduced to reduced
    const VXd applyTransformationTranspose(const VXd &ISV) const {
        return m_midSurfaceToPeriodicPatchMapTranspose_all_vars.apply(ISV, /* transpose */ false);
    }

    // The z variables are transformed so that the average z variables are exposed, but the total number of variables remains the same.
    size_t numVars() const { return ipu.numVars(); }
    // The following functions are needed for inflation_newton class to work.
    size_t numMacroFVars() const { return ipu.numMacroFVars(); }
    size_t numMacroRVars() const { return ipu.numMacroRVars(); }
    size_t numFluctuationDisplacementVars() const { return ipu.numFluctuationDisplacementVars(); }

    Real areaFactor() const { return ipu.areaFactor(); }

    std::vector<int> getRigidMotionFixedVars() {
        std::vector<int> ipu_fixedVars = ipu.get_center_fixedVars();
        return std::vector<int>({ipu_fixedVars[0], ipu_fixedVars[1], (int)get_average_z_idx()});
    }
    
    std::vector<int> getStretchingStiffnessFixedVars() {
        // We need to constrain 1. the rigid motion, 2. the average deformation gradient, 3. the bending variables
        //  to comute the equilibrium derivative of the fluctuation displacement variables on the average deformation gradient variables in planar state.
        std::vector<int> fixedVars = getRigidMotionFixedVars();
        fixedVars.insert(fixedVars.end(), {0, 1, 2, (int)numVars() - 1, (int)numVars() - 2});
        return fixedVars;
    }

    std::vector<int> getBendingStiffnessFixedVars() {
        // We need to constrain 1. the rigid motion, 2. the bending variables
        // to compute the equilibrium derivative of the average deformation gradients and the fluctuation displacement variables on the bending variables in planar state.
        std::vector<int> fixedVars = getRigidMotionFixedVars();
        fixedVars.insert(fixedVars.end(), {(int)numVars() - 1, (int)numVars() - 2});
        return fixedVars;
    }

    Real get_alpha() const { return ipu.get_alpha(); }
    SuiteSparseMatrix getHessianSquareBlock(const SuiteSparseMatrix &full_hessian, size_t nRows) {
        return ipu.getHessianSquareBlock(full_hessian, nRows);
    }

    VXd getVars() const {
        VXd base_vars = ipu.getVars();
        size_t numZVars = ipu.numFluctuationDisplacementVars() / 3;

        VXd ZVars(numZVars);
        for (size_t i = 0; i < numZVars; ++i) {
            ZVars(i) = base_vars(ipu.numMacroFVars() + i * 3 + 2);
        }
        Real average_z = ipu.get_average_z();
        VXd AZVars(numZVars);
        AZVars.setZero();
        AZVars(0) = ZVars(0) * m_ipu_vx_multiplicity(0) - average_z;
        for (size_t i = 1; i < numZVars - 1; ++i) {
            AZVars(i) = ZVars(i) * m_ipu_vx_multiplicity(i) + AZVars(i-1) - average_z;
        }
        AZVars(numZVars - 1) = average_z;
        // if (std::abs(AZVars.sum()) > 1e-10) throw std::runtime_error("The sum of z coordinates of ipu is not zero!");
        VXd derived_vars;
        derived_vars.resize(numVars());
        // Copy over the variables and then modify the fluctuation z variables.
        derived_vars.head(numVars()) = base_vars.head(numVars());
        for (size_t i = 0; i < numZVars; ++i) {
            derived_vars(ipu.numMacroFVars() + i * 3 + 2) = AZVars(i);
        }
        // derived_vars.tail(ipu.numMacroRVars()) = base_vars.tail(ipu.numMacroRVars());
        return derived_vars;
    }

    void setVars(Eigen::Ref<const VXd> vars) {
        ipu.setVars(applyTransformation(vars));
    }
    
    size_t get_average_z_idx() const { return ipu.numMacroFVars() + ipu.numFluctuationDisplacementVars() - 1; }
    
    Real get_average_z() const { return ipu.get_average_z(); }
    
    void setPressure(Real pressure) {
        ipu.sheet.setPressure(pressure);
    }

    Real energy(EnergyType etype = EnergyType::Full) const { return ipu.energy(etype); }

    Real systemEnergy() const { return energy(EnergyType::Elastic); }

    VXd gradient(EnergyType etype = EnergyType::Full) const {
        return applyTransformationTranspose(ipu.gradient(etype));
    }

    SuiteSparseMatrix baseHessianSparsityPattern() const;

    size_t hessianNNZ() const { return hessianSparsityPattern().nz; } 
    SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const;

    void              hessian(SuiteSparseMatrix &H, EnergyType etype = EnergyType::Full, bool kappaOnly = false) const; // accumulate Hessian to H
    SuiteSparseMatrix hessian(                      EnergyType etype = EnergyType::Full, bool kappaOnly = false) const; // construct and return Hessian

    Real get_kappa_second_derivative(EnergyType etype = EnergyType::Full) const {
        return ipu.get_kappa_second_derivative(etype);
    }
    SuiteSparseMatrix getMidSurfaceToPeriodicPatchMapTranspose_all_vars() const { return m_midSurfaceToPeriodicPatchMapTranspose_all_vars; }

    void setHessianColIdentity(SuiteSparseMatrix &hessian, size_t colIdx) {
        ipu.setHessianColIdentity(hessian, colIdx);
    }
    InflatablePeriodicUnit ipu;

    ////////////////////////////////////////////////////////////////////////////
    // Visualization
    ////////////////////////////////////////////////////////////////////////////

    std::shared_ptr<Mesh> visualizationMesh(bool duplicateFusedTris = false) const { 
    // Visualize (copies of) the periodic unit.
        return ipu.visualizationMesh(duplicateFusedTris); 
    }

    Eigen::MatrixXd visualizationField(Eigen::MatrixXd field, bool duplicateFusedTris = false) { return ipu.visualizationField(field, duplicateFusedTris); };


protected:

private:
    SuiteSparseMatrix m_midSurfaceToPeriodicPatchMapTranspose_all_vars;
    mutable std::unique_ptr<SuiteSparseMatrix> m_cachedHessianSparsity, m_cachedBaseHessianSparsity;
    VXd m_ipu_vx_multiplicity;

};



#endif