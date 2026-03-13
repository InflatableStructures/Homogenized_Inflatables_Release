#ifndef INFLATABLEPERIODICUNIT_HH
#define INFLATABLEPERIODICUNIT_HH

#include <MeshFEM/BoundaryConditions.hh>
#include <stdexcept>
#include "InflatableSheet.hh"
#include <cmath>
#include <rotation_optimization.hh>
// #include <MeshFEM/ElasticSolidRotExtrap.hh>
#include "MeshFEM_temp_RotExtrap.hh"
#include <MeshFEM/newton_optimizer/newton_optimizer.hh>

// Implement the periodic boundary condition through a linear change of variables. 
// The periodic boundary condition is only applied in two dimensions; 
// the vertices in the third dimension are free to move. 
// The new reduced variables are (1) the periodic fluctuation displacement variables 
// (3 * nv - 2 * (n_{bv} - 1)), and (2) the average deformation gradient. We set the 
// deformation gradient to be symmetric since our energy is rotation invariant; 
// the third dimension is fixed to identity since the fluctuation displacement without
// periodic boundary condition in the third dimention can cover everything. 
// So the average deformation gradient only have three free variables.
// List the variables using the following order:
//     f^1  f^2  f^3  u^1_x  u^1_y  u^1_z  u^2_x  u^2_y  u^2_z  u^3_x  u^3_y  u^3_z ...
// The change of variable equation is as follows:
//  X^1 0   Y^1  1 0 0 0 0 0 0... 0
//  0   Y^1 X^1  0 1 0 0 0 0 0... 0
//  0    0   0   0 0 1 0 0 0 0... 0
//  X^2 0   Y^2  0 0 0 1 0 0 0... 0
//  0   Y^2 X^2  0 0 0 0 1 0 0... 0
//  0    0   0   0 0 0 0 0 1 0... 0

struct InflatablePeriodicUnit;
struct x_hat_sensitivity; // Defined in BendingChangeVarSensitivity.hh

struct InflatablePeriodicUnit {
#if INFLATABLES_LONG_DOUBLE
    using Real = long double;
#else
    using Real = double;
#endif
    using  V3d = typename InflatableSheet::V3d;
    using  V2d = typename InflatableSheet::V2d;
    using  VXd = typename InflatableSheet::VXd;
    using MX2d = typename InflatableSheet::MX2d;
    using MX3d = typename InflatableSheet::MX3d;
    using  M3d = typename InflatableSheet::M3d;
    using  M2d = typename InflatableSheet::M2d;

    using Mesh = typename InflatableSheet::Mesh;
    using SheetEnergyType = typename InflatableSheet::EnergyType;
    using EnergyDensity = typename InflatableSheet::EnergyDensity;
    static constexpr size_t NONE = std::numeric_limits<size_t>::max();

    static constexpr size_t N = 3;
    static constexpr size_t numMacroFVars() { return 3; }

    static constexpr bool average_z_reparametrization = false;

    // Kappa, a_hat x, a_hat y

    using PCondition = PeriodicCondition<N>;
    using TMatrix = TripletMatrix<Triplet<Real>>;
    using RO = rotation_optimization<Real>;
    // Periodic unit patch of an inflatable sheet.

    // Ignore gravity in the homogenization stage.
    enum class EnergyType { Full, Elastic, Pressure };

    size_t getPeriodicIndexForSheetVx(size_t ni) const { return m_input_mesh_periodic_condition.periodicDoFsForNodes()[ni]; }

    size_t get_IPU_vidx_for_inflatable_vidx(size_t si) const {
        return m_periodicIndex_for_inflatable_vidx[si];
    }

    InflatablePeriodicUnit(const std::shared_ptr<Mesh> &inMesh, const std::vector<bool> &fusedVtx = std::vector<bool>(), Real epsilon = 1e-7);

    void setMaterial(const EnergyDensity &psi) {
        sheet.setMaterial(psi);
    }

    SuiteSparseMatrix periodicPatchToInflatableSheetMapTranspose() const { return m_periodicPatchToInflatableSheetMapTranspose; }
    void m_constructPeriodicPatchToInflatableSheetMapTranspose();

    // reduced to unreduced
    const VXd applyTransformation(const VXd &PPV) const {
        return m_periodicPatchToInflatableSheetMapTranspose.apply(PPV, /* transpose */ true);
    }
    // Unreduced to reduced
    const VXd applyTransformationTranspose(const VXd &ISV) const {
        return m_periodicPatchToInflatableSheetMapTranspose.apply(ISV, /* transpose */ false);
    }

    // planar sheet vars to unreduced
    const VXd applyBendingTransformation(const VXd &PSV) const {
        VXd result = VXd::Zero(PSV.size());

        for (size_t i = 0; i < size_t(PSV.size() / 3); ++i) {
            V3d u = PSV.segment<3>(3 * i) - get_center();
            Real h = u(2);
            u(2) = 0.0;
            result.segment<3>(3 * i) = u + get_center();

            // axis_mat << - cos(m_alpha) * sin(m_alpha), cos(m_alpha) * cos(m_alpha), 0, 
            //         - sin(m_alpha) * sin(m_alpha), cos(m_alpha) * sin(m_alpha), 0, 
            //         0, 0, 0;
            V3d omega = m_grad_omega * u;
            Real theta_sq = omega.dot(omega);
            Real theta = sqrt(theta_sq);
            result.segment<3>(3 * i) += RotExtrap<Real, 3>::apply_Rtilde(omega, u);
            // Normal terms
            result.segment<3>(3 * i) += h * RO::rotated_vector(omega, V3d::UnitZ());

            //   one_minus_cos_div_theta_sq(theta, theta_sq) * omega.cross(curr_x) + theta_minus_sin_div_theta_cubed(theta, theta_sq) * omega.cross(omega.cross(curr_x));
        }
        return result;
    }

    size_t numFluctuationDisplacementVars() const { return m_numFluctuationDisplacementVertices * N; }

    size_t                  numMacroRVars() const { return m_planar_homogenization ? 0 : 2; }
    size_t                   numMacraVars() const { return numMacroFVars() + numMacroRVars(); }

    size_t                        numVars() const { return numMacraVars() + numFluctuationDisplacementVars(); }

    VXd getVars() const {
        VXd result(numVars());
        injectMacroF(result, m_F);
        injectFluctuationDisplacements(result, m_w);
        if (!m_planar_homogenization) injectMacroR(result, m_R);
        return result;
    }

    void setIdentityDeformation() {
        m_F = V3d(1, 1, 0);
        m_R = V2d(0, 0);
        configureBendingVars();
        m_w = VXd::Zero(numFluctuationDisplacementVars());
        m_x_flat = sheet.getVars();

    }

    Real areaFactor() const {
        return m_F(0) * m_F(1) - m_F(2) * m_F(2);
    }

    std::pair<double, double> get_deformation_scale_factors() const {
        Eigen::Matrix2d mat;
        mat << m_F(0), m_F(2),
               m_F(2), m_F(1);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(mat);
        if (eigensolver.info() != Eigen::Success) {
            // Handle failure to compute eigenvalues
            throw std::runtime_error("Failed to compute eigenvalues for the average deformation gradient");
        }

        double min_eigenvalue = eigensolver.eigenvalues().minCoeff();
        double max_eigenvalue = eigensolver.eigenvalues().maxCoeff();

        return std::make_pair(min_eigenvalue, max_eigenvalue);
    }
    
    std::vector<int> get_center_fixedVars() {
        int fixedVxIdx = get_IPU_vidx_for_inflatable_vidx(sheet.center_non_fused_vx_idx());
        std::vector<int> fixedVars;
        for (int i = 0; i < 3; ++i) {
            fixedVars.push_back(3 + fixedVxIdx * 3 + i);
        }
        return fixedVars;
    }

    std::vector<int> getStretchingStiffnessFixedVars() {
        // We need to constrain 1. the rigid motion, 2. the average deformation gradient, 3. the bending variables
        //  to comute the equilibrium derivative of the fluctuation displacement variables on the average deformation gradient variables in planar state.
        std::vector<int> result = get_center_fixedVars();
        result.insert(result.end(), {0, 1, 2, (int)numVars() - 1, (int)numVars() - 2});
        return result;
    }

    std::vector<int> getBendingStiffnessFixedVars() {
        // We need to constrain 1. the rigid motion, 2. the bending variables
        // to compute the equilibrium derivative of the average deformation gradients and the fluctuation displacement variables on the bending variables in planar state.
        std::vector<int> result = get_center_fixedVars();
        result.insert(result.end(), {(int)numVars() - 1, (int)numVars() - 2});
        return result;
    }

    void injectMacroF(VXd &vars, const V3d &macroF) const {
        vars.head(numMacroFVars()) = macroF;
    }

    VXd extractMacroF(const VXd &vars) const {
        return vars.head(numMacroFVars());
    }

    void injectMacroR(VXd &vars, const V2d &macroR) const {
        if (m_planar_homogenization) throw std::runtime_error("Cannot inject macroR in planar homogenization mode");
        vars.tail(numMacroRVars()) = macroR;
    }

    VXd extractMacroR(const VXd &vars) const {
        return vars.tail(numMacroRVars());
    }

    auto fluctuationDisplacementView(      VXd &vars) const { return vars.segment(numMacroFVars(), numFluctuationDisplacementVars()); }
    auto fluctuationDisplacementView(const VXd &vars) const { return vars.segment(numMacroFVars(), numFluctuationDisplacementVars()); }

    void injectFluctuationDisplacements(VXd &vars, const VXd &fluctuation_displacements) const {
        fluctuationDisplacementView(vars) = fluctuation_displacements;
    }

    VXd extractFluctuationDisplacements(const VXd &vars) const {
        return fluctuationDisplacementView(vars);
    }

    void update_boundary_triangles() {
        for (auto &tri: m_boundary_triangles) tri.update(sheet);
    }

    void setVars(Eigen::Ref<const VXd> vars) {
        if (size_t(vars.size()) != numVars()) throw std::runtime_error("Wrong number of variables");
        m_F = extractMacroF(vars);
        m_w = extractFluctuationDisplacements(vars);
        if (m_planar_homogenization)
            sheet.setVars(applyTransformation(vars));
        else {
            m_R = extractMacroR(vars);
            configureBendingVars();
            m_x_flat = applyTransformation(vars).head(sheet.numVars());
            sheet.setVars(applyBendingTransformation(m_x_flat));
        }

        // Update information for boundary triangles.
        update_boundary_triangles();
        m_sensitivityCache.clear();
    }

    void reparametrize_vertical_offset();

    Real energy(EnergyType etype = EnergyType::Full) const { 
        Real normalization_factor = 1.0 / m_initial_area;
        Real energy_val = 0.0;
        if (etype == EnergyType::Full || etype == EnergyType::Elastic) {
            energy_val += sheet.energy(SheetEnergyType::Elastic);
        }
        if (etype == EnergyType::Full || etype == EnergyType::Pressure) {
            energy_val += sheet.energy(SheetEnergyType::Pressure) + energyPeriodicPressurePotential();
        }
        return normalization_factor * energy_val;
    }

    Real systemEnergy() const { return energy(EnergyType::Elastic); }

    VXd getSheetGradient(EnergyType etype) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("InflatablePeriodicUnit.getSheetGradient");
        VXd sheet_gradient = VXd::Zero(sheet.numVars());
        if (etype == EnergyType::Full || etype == EnergyType::Elastic) {
            sheet_gradient += sheet.gradient(SheetEnergyType::Elastic);
        }
        if (etype == EnergyType::Full || etype == EnergyType::Pressure) {
            sheet_gradient += sheet.gradient(SheetEnergyType::Pressure, false);
            sheet_gradient += gradientPeriodicPressurePotential();
        }

        return sheet_gradient / m_initial_area;
    }

    VXd gradient(EnergyType etype = EnergyType::Full) const {
        if (m_planar_homogenization) {
            return applyTransformationTranspose(getSheetGradient(etype));
        } else {
            return applyTransformationTranspose(bent_sheet_gradient(etype));
        }
    }

    size_t hessianNNZ() const { return hessianSparsityPattern().nz; } 
    SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const;

    void              hessian(SuiteSparseMatrix &H, EnergyType etype = EnergyType::Full, bool kappaOnly = false) const; // accumulate Hessian to H
    SuiteSparseMatrix hessian(                      EnergyType etype = EnergyType::Full, bool kappaOnly = false) const; // construct and return Hessian

    void baseHessian(SuiteSparseMatrix &baseH, EnergyType etype = EnergyType::Full) const;
    SuiteSparseMatrix baseHessian(             EnergyType etype = EnergyType::Full) const; // construct and return Hessian

    Real get_kappa_second_derivative(EnergyType etype = EnergyType::Full) const {
        SuiteSparseMatrix H = bent_sheet_hessian(etype, true);
        return H.Ax[H.findDiagEntry(sheet.numVars())];
    }

    void setPressure(Real pressure) {
        sheet.setPressure(pressure);
    }
    ////////////////////////////////////////////////////////////////////////////
    // Bending homogenization
    ////////////////////////////////////////////////////////////////////////////
    size_t bent_sheet_numVars() const {
        return sheet.numVars() + 2;
    }

    // Debug only.
    VXd bent_sheet_getVars() const {
        VXd result(sheet.numVars() + 2);
        result.head(sheet.numVars()) = m_x_flat;
        result.tail(2) = m_R;
        return result;
    }

    // Debug only.
    void bent_sheet_setVars(Eigen::Ref<const VXd> vars) {
        m_R = vars.tail(2);
        configureBendingVars();
        m_x_flat = vars.head(sheet.numVars());
        sheet.setVars(applyBendingTransformation(m_x_flat));
        // Update information for boundary triangles.
        update_boundary_triangles();
        m_sensitivityCache.clear();

    }

    // The gradient of the energy over sheet variables configured by the bending variables but without periodic boundary conditions. 
    VXd bent_sheet_gradient(EnergyType etype = EnergyType::Full) const;

    SuiteSparseMatrix bentSheetHessianSparsityPattern(Real val = 0.0) const;

    void   bent_sheet_hessian(SuiteSparseMatrix &H, EnergyType etype = EnergyType::Full, bool kappaOnly = false) const;
    SuiteSparseMatrix bent_sheet_hessian(           EnergyType etype = EnergyType::Full, bool kappaOnly = false) const; // construct and return Hessian

    ////////////////////////////////////////////////////////////////////////////
    // Periodic boundary conditions
    ////////////////////////////////////////////////////////////////////////////

    SuiteSparseMatrix getPeriodicPatchToInflatableSheetMapTranspose() const { return m_periodicPatchToInflatableSheetMapTranspose; }

    InflatableSheet sheet;

    // We only need half of the boundary for each axis to compute the missing volume term due to the periodic boundary.
    bool checkBoundaryEdgeOnMinXY(size_t v0, size_t v1) const {
        return ((m_input_mesh_periodic_condition.bdryNodeOnMinOrMaxPeriodCellFace(v0,  0) == -1 || m_input_mesh_periodic_condition.bdryNodeOnMinOrMaxPeriodCellFace(v0,  1) == -1) &&
                (m_input_mesh_periodic_condition.bdryNodeOnMinOrMaxPeriodCellFace(v1,  0) == -1 || m_input_mesh_periodic_condition.bdryNodeOnMinOrMaxPeriodCellFace(v1,  1) == -1));
    }

    // Whether the edge connecting vertex indices `v0` and `v1` contributes a
    // periodic volume term to the pressure potential.
    bool edgeContributesPeriodicVolume(size_t v0, size_t v1) const {
        // Fused edges enclose no volume.
        if (sheet.isWallVtx(v0) && sheet.isWallVtx(v1)) return false;

        // Each identified pair of periodic boundary faces contributes a term;
        // we arbitrarily pick the "min" face as a representative for each pair,
        // contributing volume only for it.
        auto &pc = getPeriodicCondition();
        size_t bv0 = sheet.mesh().vertex(v0).boundaryVertex().index();
        size_t bv1 = sheet.mesh().vertex(v1).boundaryVertex().index();
        return pc.periodicBoundariesForBoundaryNode(bv0).onAnyMinFace() // Assumes mesh doesn't contain an unfused edge connecting nodes on different periodic boundaries...
            && pc.periodicBoundariesForBoundaryNode(bv1).onAnyMinFace();
    }

    // For the boundary edge connecting vertices v0 and v1, determine which (if
    // any) is a corner, and an arbitrary non-corner (singly-paired) vertex.
    struct BECornerInfo {
        size_t cornerVtx = NONE, noncornerVtx = NONE;
        bool incidentCorner() const { return cornerVtx != NONE; }
    };

    struct BoundaryTriangle {
        BoundaryTriangle(std::vector<size_t> inputVxIdx, std::vector<size_t> inputSheetIdx) : vxIdx(inputVxIdx), sheetIdx(inputSheetIdx) {
            if (vxIdx.size() != 3) throw std::runtime_error("BoundaryTriangle must have 3 vertices!");
            if (sheetIdx.size() != 3) throw std::runtime_error("BoundaryTriangle must have 3 sheet indices!");
        }
        std::vector<size_t> vxIdx;
        std::vector<size_t> sheetIdx;
        M3d triCornerPos;
        V3d deformed_normal;
        Real deformed_area;
        V3d deformed_normal_scaled_by_area;
        void update(const InflatableSheet &deformed_sheet) {
            for (size_t i = 0; i < 3; i++) {
                triCornerPos.col(i) = deformed_sheet.getDeformedVtxPosition(vxIdx[i], sheetIdx[i]);
            }
            const V3d n = (triCornerPos.col(1) - triCornerPos.col(0)).cross(triCornerPos.col(2) - triCornerPos.col(0));
            const Real dblA = n.norm();
            deformed_area = 0.5 * dblA;
            deformed_normal = n / dblA;
            deformed_normal_scaled_by_area = 0.5 * n;
        }
    };
    BECornerInfo identifyCorner(size_t v0, size_t v1) const {
        BECornerInfo result;
        if (getPeriodicCondition().hasSinglePair(v0)) { result.noncornerVtx = v0; } else { result.cornerVtx = v0; }
        if (getPeriodicCondition().hasSinglePair(v1)) { result.noncornerVtx = v1; } else { result.cornerVtx = v1; }
        if (result.noncornerVtx == NONE) throw std::runtime_error("Each edge must contain at least one non-corner vertex (mesh too coarse)!");
        return result;
    }

    V3d getOppoVxPosition(size_t vtxIdx, size_t sheetIdx) const {
        return sheet.getDeformedVtxPosition(getOppoVxIdx(vtxIdx), sheetIdx);
    }

    size_t getOppoVxIdx(size_t vtxIdx) const {
        return getPeriodicCondition().pairedNode(vtxIdx);
    }

    // The outputs are *boundary* nodes.
    std::tuple<size_t, size_t> pairedEdge(size_t _vni, size_t _vnj) const;

    // Refactor this code to be separated between .hh and .cc files.
    // Missing enclosed volume from the periodic boundary.
    Real periodicVolume() const;
    
    Real energyPeriodicPressurePotential() const {
        return -periodicVolume() * sheet.getPressure();
    }

    // The gradient here is over the original inflatable sheet variables.
    VXd gradientPeriodicPressurePotential() const;

    // The hessian here is over the original inflatable sheet variables.
    template <typename MatrixType>
    void hessianPeriodicPressurePotential(MatrixType &Hout) const;

    TMatrix hessianPeriodicPressurePotential() const {
        TMatrix result(sheet.numVars(), sheet.numVars());
        result.symmetry_mode = TMatrix::SymmetryMode::UPPER_TRIANGLE;
        hessianPeriodicPressurePotential(result);
        return result;
    }

    void addPeriodicVolumeSparsityPattern(SuiteSparseMatrix &Hsp, Real val) const;

    SuiteSparseMatrix baseHessianSparsityPattern() const;

    Real getPeriodicUnitDimension(size_t axis) const {
        return m_inputMeshDimension[axis] * m_F(axis);
    }
    const PCondition &getPeriodicCondition() const { return m_input_mesh_periodic_condition; }

    SuiteSparseMatrix getHessianSquareBlock(const SuiteSparseMatrix &full_hessian, size_t nRows) {
        SuiteSparseMatrix hessian_block(nRows, nRows);
        hessian_block.symmetry_mode = full_hessian.symmetry_mode;
        // Build hessian_F_u by extracting the first num_Fu columns.
        hessian_block.Ap.resize(nRows + 1);
        std::copy_n(full_hessian.Ap.begin(), nRows + 1, hessian_block.Ap.begin());
        size_t nz = hessian_block.Ap.back();
        hessian_block.Ai.resize(nz);
        hessian_block.Ax.resize(nz);
        std::copy_n(full_hessian.Ai.begin(), nz, hessian_block.Ai.begin());
        std::copy_n(full_hessian.Ax.begin(), nz, hessian_block.Ax.begin());
        return hessian_block;
    }

    void setHessianColIdentity(SuiteSparseMatrix &hessian, size_t colIdx) {
        for (size_t i = (size_t)hessian.Ap[colIdx]; i < (size_t)hessian.Ap[colIdx + 1]; i++) {
            if ((size_t)hessian.Ai[i] == colIdx)
                hessian.Ax[i] = 1.0;
            else
                hessian.Ax[i] = 0.0;
        }
    }

    template <typename MatrixType>
    void computeAHA(MatrixType &Hout, const SuiteSparseMatrix &Hin, const SuiteSparseMatrix &A) const;

    ////////////////////////////////////////////////////////////////////////////
    // Visualization
    ////////////////////////////////////////////////////////////////////////////

    void setVisualizationTilePower(size_t power) {
        visualizationTilePower = power;
    }

    size_t getVisualizationTilePower() const {
        return visualizationTilePower;
    }

    std::shared_ptr<Mesh> visualizationMesh(bool duplicateFusedTris = false) const;

    Eigen::MatrixXd visualizationField(Eigen::MatrixXd field, bool duplicateFusedTris = false) { return sheet.visualizationField(field, duplicateFusedTris); };

    VXd get_x_flat() const { return m_x_flat; }
    Real get_average_z() const {
        // Compute the average of the z coordinates from x flat.
        size_t nv = m_x_flat.size() / 3;
        Real sum = 0.0;
        for (size_t i = 0; i < nv; i++) {
            sum += m_x_flat(3 * i + 2);
        }
        return sum / nv;
    }
    Real get_kappa() const { return m_kappa; }
    Real get_alpha() const { return m_alpha; }
    M3d get_grad_omega() const { return m_grad_omega; }
    M3d get_axis_mat() const { return m_axis_mat; }

    V3d get_center() const { return m_center; }

    void set_kappa(Real kappa) {
        VXd curr_vars = getVars();
        curr_vars(numMacroFVars() + numFluctuationDisplacementVars()) = kappa;
        setVars(curr_vars);
    }
    void set_alpha(Real alpha) {
        VXd curr_vars = getVars();
        curr_vars(numMacroFVars() + numFluctuationDisplacementVars() + 1) = alpha;
        setVars(curr_vars);
    }

    bool get_use_planar_homogenization() const { return m_planar_homogenization; }
    void deactivate_planar_homogenization() { 
        m_planar_homogenization = false;
        m_constructPeriodicPatchToInflatableSheetMapTranspose();

        m_R = V2d(0, 0);
        configureBendingVars();
        m_x_flat = sheet.getVars();

        m_cachedHessianSparsity = nullptr;
        m_cachedBaseHessianSparsity = nullptr;
        m_cachedBentSheetHessianSparsity = nullptr;
        m_sensitivityCache.clear();
    }
protected:

    struct SensitivityCache {
        SensitivityCache();

        // Cache of x_hat' Jacobians and Hessians
        // (to accelerate repeated calls to elastic energy Hessian/gradient).
        std::vector<x_hat_sensitivity> sensitivityForBending;
        void update(const InflatablePeriodicUnit &ipu);
        bool filled() const { return !sensitivityForBending.empty(); }
        const x_hat_sensitivity &lookup(size_t vi) const;

        void clear();
        ~SensitivityCache();
    };
    mutable SensitivityCache m_sensitivityCache;

private:
    V3d m_F;
    V2d m_R;
    VXd m_w;
    VXd m_x_flat;

    Real m_kappa, m_alpha;
    V3d m_axis, m_axisPerp, m_z;
    M3d m_grad_omega;
    M3d m_axis_mat;

    const V3d m_center = V3d(0.0, 0.0, 0.0);
    
    void configureBendingVars() {
        m_kappa = m_R(0);
        m_alpha = m_R(1);
        m_axis << cos(m_alpha), sin(m_alpha), 0.0;
        m_axisPerp << -sin(m_alpha), cos(m_alpha), 0.0;
        m_z << 0.0, 0.0, 1.0;
        m_grad_omega = m_kappa * m_axis * m_axisPerp.transpose();
        m_axis_mat = m_axis * m_axisPerp.transpose();
    }

    // The Z coordinates of the rest state of the mesh. Currently unused because the rest state is assumed to be on the xy plane.
    // VXd m_Z;

    size_t m_numFluctuationDisplacementVertices = 0;

    PCondition m_input_mesh_periodic_condition;

    SuiteSparseMatrix m_periodicPatchToInflatableSheetMapTranspose, m_sparse_periodicPatchToInflatableSheetMapTranspose, m_sparse_periodicPatchToInflatableSheetMap;
    MX3d m_d_x_d_F;
    
    mutable std::unique_ptr<SuiteSparseMatrix> m_cachedHessianSparsity, m_cachedBaseHessianSparsity, m_cachedBentSheetHessianSparsity;

    std::vector<size_t> m_periodicIndex_for_inflatable_vidx;
    std::vector<size_t> m_inputMeshIndex_for_inflatable_vidx;

    std::vector<Real> m_inputMeshDimension;

    std::vector<BoundaryTriangle> m_boundary_triangles;

    size_t visualizationTilePower = 0;

    bool m_planar_homogenization = false;

    // Spin locks used for parallel Hessian assembly.
    mutable std::unique_ptr<std::vector<std::atomic<bool>>> m_varLocks;
    auto &m_getVarLocks() const {
        if (!m_varLocks) {
            const size_t nv = numVars();
            m_varLocks = std::make_unique<std::vector<std::atomic<bool>>>(nv);
            for (size_t i = 0; i < nv; ++i)
                atomic_init(&(*m_varLocks)[i], false);
        }
        return *m_varLocks;
    }

    Real m_initial_area = 1.0;
};

#endif