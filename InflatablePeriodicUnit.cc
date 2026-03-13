#include "InflatablePeriodicUnit.hh"
#include "MeshFEM/GlobalBenchmark.hh"
#include "BendingChangeVarSensitivity.hh"

InflatablePeriodicUnit::InflatablePeriodicUnit(const std::shared_ptr<Mesh> &inMesh, const std::vector<bool> &fusedVtx, Real epsilon) 
    : sheet(inMesh, fusedVtx), m_input_mesh_periodic_condition(PCondition(*inMesh, epsilon, false, std::vector<size_t>{2})) { 
    // Using the area of the unit cell to normalize the energy. 
    m_initial_area = inMesh->boundingBox().dimensions()[0] * inMesh->boundingBox().dimensions()[1];

    {

        m_periodicIndex_for_inflatable_vidx.resize(sheet.numVars() / 3);
        m_inputMeshIndex_for_inflatable_vidx.resize(sheet.numVars() / 3);
        // Inflatable periodic unit vertex index for the (duplicated) input sheet periodic vertex index. 
        // Some of the periodic vertex in the input sheet are never queried or stored, e.g. the fused vertex in the bottom sheet.
        // The first index are the periodic index in the sheet; the second index is for the index of the sheet: 0 for top sheet and 1 for bottom sheet.
        Eigen::Matrix<int, Eigen::Dynamic, 2> m_IPU_vidx_for_ISP_vidx;
        m_IPU_vidx_for_ISP_vidx.resize(m_input_mesh_periodic_condition.numPeriodicDoFs(), 2);
        m_IPU_vidx_for_ISP_vidx.fill(-1);

        size_t newFluctuationIndex = 0;
        for (size_t si = 0; si < sheet.numVars() / 3; ++si) {
            InflatableSheet::ISheetVtx iSheetVtx = sheet.vtxForVar(si * 3);
            if (iSheetVtx.sheet <= 0) throw std::runtime_error("InflatablePeriodicUnit: couldn't find sheet vertex for variable");

            // See documentation in InflatableSheet.hh: 1 or 3 are stored in the top sheet; 2 is stored in the bottom sheet.
            size_t periodicIndex = getPeriodicIndexForSheetVx(iSheetVtx.vi);
            size_t sheetIndex = (iSheetVtx.sheet == 2) ? 1 : 0;

            m_inputMeshIndex_for_inflatable_vidx[si] = iSheetVtx.vi;
            // Generate a new fluctuation displacement variable if it's not identified to any existing variable.
            if (m_IPU_vidx_for_ISP_vidx(periodicIndex, sheetIndex) == -1) {
                m_IPU_vidx_for_ISP_vidx(periodicIndex, sheetIndex) = newFluctuationIndex;
                m_periodicIndex_for_inflatable_vidx[si] = newFluctuationIndex;
                ++ newFluctuationIndex;
            } else {
                // the periodic node has already been added
                m_periodicIndex_for_inflatable_vidx[si] = m_IPU_vidx_for_ISP_vidx(periodicIndex, sheetIndex);
            }
        }
        m_numFluctuationDisplacementVertices = newFluctuationIndex;
    }

    m_constructPeriodicPatchToInflatableSheetMapTranspose();

    setIdentityDeformation();

    m_inputMeshDimension = std::vector<Real>({sheet.mesh().boundingBox().dimensions()[0], sheet.mesh().boundingBox().dimensions()[1]});

    {
        // Initialize boundary triangles
        // These triangles are used to compute the enclosed volume and its derivatives from the periodic boundary.
        m_boundary_triangles.reserve(sheet.mesh().numBoundaryElements() * 2);
        for (const auto bhe : sheet.mesh().boundaryElements()) {
            size_t v0 = bhe.tail().volumeVertex().index();
            size_t v1 = bhe.tip() .volumeVertex().index();
            if (!edgeContributesPeriodicVolume(v0, v1)) continue;
            // Connect the top and bottom edge. Mirror the mesh on the periodic boundary.
            std::tuple<size_t, size_t> oppo_edge = pairedEdge(v0, v1);
            size_t v0_oppo = std::get<0>(oppo_edge);
            size_t v1_oppo = std::get<1>(oppo_edge);
            if (!sheet.isWallVtx(v0)) {
                m_boundary_triangles.emplace_back(std::vector<size_t>({v0, v1, v0}), std::vector<size_t>({0, 0, 1}));
                m_boundary_triangles.back().update(sheet);
                m_boundary_triangles.emplace_back(std::vector<size_t>({v1_oppo, v0_oppo, v0_oppo}), std::vector<size_t>({0, 0, 1}));
                m_boundary_triangles.back().update(sheet);
            }

            if (!sheet.isWallVtx(v1)) {
                m_boundary_triangles.emplace_back(std::vector<size_t>({v1, v1, v0}), std::vector<size_t>({0, 1, 1}));
                m_boundary_triangles.back().update(sheet);
                m_boundary_triangles.emplace_back(std::vector<size_t>({v0_oppo, v1_oppo, v1_oppo}), std::vector<size_t>({1, 1, 0}));
                m_boundary_triangles.back().update(sheet);
            }
        }
    }
}

// Reparametrize the patches to have zero average vertical offset to simplify the stiffness computation later.
void InflatablePeriodicUnit::reparametrize_vertical_offset() {
    double gamma = get_average_z();
    // New kappa.
    double kappa_gamma = m_kappa;
    double kappa_zero = kappa_gamma / (1. - kappa_gamma * gamma);

    double alpha_gamma = m_alpha;
    Eigen::Vector2d axis_perp;
    axis_perp << -sin(alpha_gamma), cos(alpha_gamma);

    Eigen::Matrix2d F_mat = Eigen::Matrix2d::Zero();
    F_mat.diagonal() = m_F.head<2>();
    F_mat(0, 1) = F_mat(1, 0) = m_F(2);

    Eigen::Matrix2d scale_factor_mat = Eigen::Matrix2d::Identity() - gamma * kappa_gamma * axis_perp * axis_perp.transpose();
    // F_zero_general is not symmetric in general. Apply polar decomposition to get a symmetric F_zero.
    Eigen::BDCSVD<Eigen::Matrix2d> svd(scale_factor_mat * F_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d rotation = svd.matrixU() * svd.matrixV().transpose();
    Eigen::Matrix2d F_zero = svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();

    double angle = atan2(rotation(1, 0), rotation(0, 0));
    double alpha_zero = alpha_gamma - angle;

    Eigen::Matrix3d inverse_rotation;
    inverse_rotation << cos(-angle), -sin(-angle), 0, sin(-angle), cos(-angle), 0, 0, 0, 1;
    // Remove gamma from z coordinates of the fluctuation displacements.
    VXd curr_vars = getVars();
    for (size_t i = 0; i < numFluctuationDisplacementVars() / 3; ++i) {
        curr_vars(numMacroFVars() + 3 * i + 2) -= gamma;
    }
    // Reshape curr_vars to numFluctuationDisplacementVars() / 3 x 3 matrix.
    // First initialize fluctuation to zeros.
    MX3d fluctuation = MX3d::Zero(numFluctuationDisplacementVars() / 3, 3);
    for (size_t i = 0; i < numFluctuationDisplacementVars() / 3; ++i) {
        fluctuation.row(i) = curr_vars.segment(numMacroFVars() + 3 * i, 3);
    }
    Eigen::Matrix3d scale_factor_sym_3D = Eigen::Matrix3d::Identity();
    scale_factor_sym_3D.topLeftCorner<2,2>() = scale_factor_mat;
    fluctuation = (scale_factor_sym_3D * fluctuation.transpose()).transpose();
    fluctuation = (inverse_rotation * fluctuation.transpose()).transpose();


    if (!(abs(F_zero(0, 1) - F_zero(1, 0)) < 1e-6)) throw std::runtime_error("F_zero is not symmetric!");

    curr_vars.head<3>() << F_zero(0, 0), F_zero(1, 1), F_zero(0, 1);
    curr_vars.tail<2>() << kappa_zero, alpha_zero;

    for (size_t i = 0; i < numFluctuationDisplacementVars() / 3; ++i) {
        curr_vars.segment(numMacroFVars() + 3 * i, 3) = fluctuation.row(i);
    }

    setVars(curr_vars);
}

void InflatablePeriodicUnit::m_constructPeriodicPatchToInflatableSheetMapTranspose() {
    const SuiteSparse_long m = numMacraVars() + numFluctuationDisplacementVars(), n = sheet.numVars() + numMacroRVars();
    SuiteSparseMatrix result(m, n);
    result.nz = sheet.numVars() / 3 * 7 + numMacroRVars();

    SuiteSparseMatrix sparse_result(m, n);
    sparse_result.nz = sheet.numVars() + numMacroRVars();

    MX3d d_x_d_F;
    d_x_d_F.resize(n, 3);
    d_x_d_F.setZero();

    // Now we fill out the transpose of the map one column (arm segment) at a time:
    //         3         [                                   ]
    //     3 * # IPU vxs [                                   ]
    //  2 (if not planar)[                                   ]
    //                      3 * # IS vxs + 2 (if not planar)

    auto &Ai = result.Ai;
    auto &Ap = result.Ap;
    auto &Ax = result.Ax;

    auto &sparse_Ai = sparse_result.Ai;
    auto &sparse_Ap = sparse_result.Ap;
    auto &sparse_Ax = sparse_result.Ax;

    Ai.reserve(result.nz);
    Ap.reserve(n + 1);

    Ap.push_back(0); // col 0 begin

    sparse_Ai.reserve(sparse_result.nz);
    sparse_Ap.reserve(n + 1);
    sparse_Ap.push_back(0); // col 0 begin

    for (size_t si = 0; si < sheet.numVars() / 3; ++si) {

        Real rest_X = sheet.getVars()(3 * si);
        Real rest_Y = sheet.getVars()(3 * si + 1);
        size_t ipu_vidx = get_IPU_vidx_for_inflatable_vidx(si);

        // Explicitly expand this loop to read the matrix.
        for (size_t ci = 0; ci < 3; ++ci) {
            if (ci < 2) {
                Ai.push_back(2 * ci);
                Ai.push_back(2 - ci);

                Ax.push_back(rest_X);
                Ax.push_back(rest_Y);

                d_x_d_F(3 * si + ci, 2 * ci) = rest_X;
                d_x_d_F(3 * si + ci, 2 - ci) = rest_Y;
            }

            Ai.push_back(3 + ipu_vidx * 3 + ci);
            Ax.push_back(1);
            Ap.push_back(Ai.size()); // col end

            sparse_Ai.push_back(3 + ipu_vidx * 3 + ci);
            sparse_Ax.push_back(1);
            sparse_Ap.push_back(sparse_Ai.size()); // col end

        }
    }

    for (size_t ri = 0; ri < numMacroRVars(); ++ri) {
        Ai.push_back(numMacroFVars() + numFluctuationDisplacementVars() + ri);
        Ax.push_back(1);
        Ap.push_back(Ai.size()); // col end

        sparse_Ai.push_back(numMacroFVars() + numFluctuationDisplacementVars() + ri);
        sparse_Ax.push_back(1);
        sparse_Ap.push_back(sparse_Ai.size()); // col end

    }

    if (Ai.size() != size_t(result.nz)) throw std::runtime_error("InflatablePeriodicUnit: Ai.size() != result.nz");
    if (Ap.size() != size_t(n + 1    )) throw std::runtime_error("InflatablePeriodicUnit: Ap.size() != n + 1");

    if (sparse_Ai.size() != size_t(sparse_result.nz)) throw std::runtime_error("InflatablePeriodicUnit: sparse_Ai.size() != sparse_result.nz");
    if (sparse_Ap.size() != size_t(n + 1    )) throw std::runtime_error("InflatablePeriodicUnit: sparse_Ap.size() != n + 1");

    m_periodicPatchToInflatableSheetMapTranspose = std::move(result);
    m_sparse_periodicPatchToInflatableSheetMapTranspose = std::move(sparse_result);
    m_sparse_periodicPatchToInflatableSheetMap = m_sparse_periodicPatchToInflatableSheetMapTranspose.transpose();
    m_d_x_d_F = std::move(d_x_d_F);
}

// Compute Hout = A^T Hin A, where Hin  is a symmetric matrix in upper triangular format, it can either be CSC or Triplet format.
//                                 A    is also in CSC format, but it's not necessarily square or symmetric.
//                                 Hout is in upper triangular CSC format.
template <typename MatrixType>
void InflatablePeriodicUnit::computeAHA(MatrixType &Hout, const SuiteSparseMatrix &Hin, const SuiteSparseMatrix &A) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("InflatablePeriodicUnit.computeAHA");
    size_t hint = 0;
    for (size_t j = 0; j < size_t(Hin.n); ++j) {
        // Loop over each output column "l" generated by angle variable "j"
        const size_t lend = A.Ap[j + 1];
        for (size_t idx = A.Ap[j]; idx < lend; ++idx) {
            const size_t l = A.Ai[idx];
            const Real colMultiplier = A.Ax[idx];

            // Create entries for each input Hessian entry
            const size_t input_end = Hin.Ap[j + 1];
            for (size_t idx_in = Hin.Ap[j]; idx_in < input_end; ++idx_in) {
                const Real colVal = colMultiplier * Hin.Ax[idx_in];
                const size_t i = Hin.Ai[idx_in];
                // Loop over each output entry
                const size_t outrow_end = A.Ap[i + 1];
                for (size_t outrow_idx = A.Ap[i]; outrow_idx < outrow_end; ++outrow_idx) {
                    const size_t k = A.Ai[outrow_idx];
                    const Real val = A.Ax[outrow_idx] * colVal;
                    if (k <= l) {
                        // Accumulate entries from input's upper triangle
                        hint = Hout.template addNZAtLoc<false>(k, l, val, hint);
                    }
                    if ((i != j) && (l <= k)) Hout.template addNZAtLoc<false>(l, k, val, hint); // accumulate entries from input's (strict) lower triangle
                }
            }
        }
    }
}

// The outputs are *boundary* nodes.
std::tuple<size_t, size_t> InflatablePeriodicUnit::pairedEdge(size_t _vni, size_t _vnj) const {
    BECornerInfo ci = identifyCorner(_vni, _vnj);
    if (ci.incidentCorner()) {
        const auto &inodes = getPeriodicCondition().identifiedNodesInBoundaryNodeIndex(ci.cornerVtx);
        const size_t bnc = getPeriodicCondition().getBoundaryNodeIndexForNode(ci.cornerVtx);
        int bnc_0 = getPeriodicCondition().bdryNodeOnMinOrMaxPeriodCellFace(bnc, 0);
        int bnc_1 = getPeriodicCondition().bdryNodeOnMinOrMaxPeriodCellFace(bnc, 1);
        // A non-corner vertex will be on only one boundary face.
        int target_0 = -bnc_0; int target_1 = bnc_1;
        if (getPeriodicCondition().bdryNodeOnMinOrMaxPeriodCellFace(getPeriodicCondition().getBoundaryNodeIndexForNode(ci.noncornerVtx), 0) == 0) { target_0 = bnc_0; target_1 = -bnc_1; }
        // Then the opposite of bnj should be on the opposite in dimension 1 but same as bnj in dimension 0
        for (size_t i = 0; i < 4; ++i) {
            if (getPeriodicCondition().bdryNodeOnMinOrMaxPeriodCellFace(inodes[i], 0) == target_0 &&
                getPeriodicCondition().bdryNodeOnMinOrMaxPeriodCellFace(inodes[i], 1) == target_1) {
                if (ci.noncornerVtx == _vni) return std::make_tuple(getOppoVxIdx(ci.noncornerVtx), sheet.mesh().boundaryNode(inodes[i]).volumeNode().index());
                else return std::make_tuple(sheet.mesh().boundaryNode(inodes[i]).volumeNode().index(), getOppoVxIdx(ci.noncornerVtx));
            }
        }
        throw std::runtime_error("Couldn't find pair edges!");
    } else {
        return std::make_tuple<size_t, size_t> (getOppoVxIdx(_vni), getOppoVxIdx(_vnj));
    }
}

// Missing enclosed volume from the periodic boundary.
Real InflatablePeriodicUnit::periodicVolume() const {
    InflatableSheet::NeumaierSum<Real> sum(0);
    if (!m_planar_homogenization) {
        for (const auto &tri : m_boundary_triangles) {
            sum.accumulate(tri.triCornerPos.determinant());
        }
    } else {
        // We can compute these from the boundary curves only when the sheet is planar. 
        // But we could also use the same formula above for non-planar sheets. 
        // These are just kept from the old code.
        for (const auto bhe : sheet.mesh().boundaryElements()) {
            size_t v0_idx = bhe.tail().volumeVertex().index();
            size_t v1_idx = bhe.tip() .volumeVertex().index();
            if (!edgeContributesPeriodicVolume(v0_idx, v1_idx)) continue;
            BECornerInfo ci = identifyCorner(v0_idx, v1_idx);

            // Rigid translation relating the two period cell boundary faces containing `bhe`.
            V3d t = sheet.getDeformedVtxPosition(ci.noncornerVtx, 0)
                            - getOppoVxPosition(ci.noncornerVtx, 0);

            Real edgeContrib = 0;
            // Top sheet then bottom sheet.
            for (int sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
                edgeContrib -= getOppoVxPosition(ci.noncornerVtx, sheetIdx).dot(sheet.getDeformedVtxPosition(v0_idx, sheetIdx)
                            .cross(sheet.getDeformedVtxPosition(v1_idx, sheetIdx)));
                std::swap(v0_idx, v1_idx); // Bottom sheet's boundary has the opposite orientation.
            }

            // Close the period cell boundary face with an edge connecting the top- and
            // bottom-sheet copies of the corner vertex.
            if (ci.incidentCorner() && !sheet.isWallVtx(ci.cornerVtx)) {
                // Determine if this "corner edge" has the top sheet at its tail.
                // This is the case if the top sheet boundary halfedge is incoming.
                bool topSheetIsTail = (ci.cornerVtx == v1_idx);
                V3d c0_pos = sheet.getDeformedVtxPosition(ci.cornerVtx, !topSheetIsTail); // tail
                V3d c1_pos = sheet.getDeformedVtxPosition(ci.cornerVtx,  topSheetIsTail); // tip
                edgeContrib += t.dot(c0_pos.cross(c1_pos));
            }
            sum.accumulate(edgeContrib);
        }
    }
    return sum.result() / 6.0;
}

// Gradient with respect to original inflatable sheet variables.
InflatablePeriodicUnit::VXd InflatablePeriodicUnit::gradientPeriodicPressurePotential() const {
    VXd result(InflatablePeriodicUnit::VXd::Zero(sheet.numVars()));
    for (const auto tri :m_boundary_triangles) {
        V3d contrib = (-sheet.getPressure() * tri.deformed_normal_scaled_by_area / 3.0);
        for (size_t i = 0; i < 3; ++i) {
            result.segment<3>(sheet.varIdx(tri.sheetIdx[i], tri.vxIdx[i])) += contrib;
        }
    }
    return result;

    // equivalent version derived more directly from the signed volume pressure potential
    // for (const auto &tri : m_boundary_triangles) {

    //     const double signed_pressure_div_6 =  sheet.getPressure() / 6.0;
    //     M3d triCornerPos = tri.triCornerPos;
    //     for (size_t i = 0; i < 3; ++i) {
    //         result.segment<3>(sheet.varIdx(tri.sheetIdx[i], tri.vxIdx[i])) -=
    //             signed_pressure_div_6 * triCornerPos.col((i + 1) % 3)
    //                                 .cross(triCornerPos.col((i + 2) % 3));
    //     }
    // }
    // return result;

}

// Hessian with respect to original inflatable sheet variables.
template <typename MatrixType>
void InflatablePeriodicUnit::hessianPeriodicPressurePotential(MatrixType &Hout) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("InflatablePeriodicUnit.hessianPeriodicPressurePotential");

    // The output hessian is only the upper triangular part.
    auto addHBlock = [&](size_t idx0, size_t idx1, M3d contrib) {
        if   (idx0 <= idx1) Hout.addNZBlock(idx0, idx1, contrib);
        else                Hout.addNZBlock(idx1, idx0, contrib.transpose().eval());
    };

    // auto &varLocks = m_getVarLocks();
    // auto assemblePerTriContrib = [&](const size_t ti) {
    for (auto &tri : m_boundary_triangles) {
        M3d triCornerPos = tri.triCornerPos;
        const double pressure_div_6 = sheet.getPressure() / 6.0;
        for (size_t vlb = 0; vlb < 3; ++vlb) {
            size_t b = sheet.varIdx(tri.sheetIdx[vlb], tri.vxIdx[vlb]);
            for (size_t vla = 0; vla < vlb; ++ vla) {
                size_t a = sheet.varIdx(tri.sheetIdx[vla], tri.vxIdx[vla]);
                const size_t vlother = 3 - (vla + vlb);
                const double ordering_sign = (vlb == ((vla + 1) % 3)) ? -1.0 : 1.0;
                V3d other_vector = (-pressure_div_6 * ordering_sign) * triCornerPos.col(vlother);
                M3d contrib = RO::cross_product_matrix(other_vector);
                addHBlock(a, b, contrib);
            }
        }
    }
    // ;

    // get_hessian_assembly_arena().execute([&assemblePerTriContrib, this]() {
    //     parallel_for_range(m_boundary_triangles.size(), assemblePerTriContrib);
    // });
}

void InflatablePeriodicUnit::addPeriodicVolumeSparsityPattern(SuiteSparseMatrix &Hsp, Real val) const {
    TMatrix base_triplet_Hsp = Hsp.getTripletMatrix();
    base_triplet_Hsp.pruneTol = -1.0;
    hessianPeriodicPressurePotential(base_triplet_Hsp);
    Hsp = SuiteSparseMatrix(base_triplet_Hsp);
    Hsp.fill(val);
}

SuiteSparseMatrix InflatablePeriodicUnit::baseHessianSparsityPattern() const {
    BENCHMARK_SCOPED_TIMER_SECTION("InflatablePeriodicUnit::baseHessianSparsityPattern");
    if (m_cachedBaseHessianSparsity) return *m_cachedBaseHessianSparsity;
    auto baseHsp = sheet.hessianSparsityPattern();
    addPeriodicVolumeSparsityPattern(baseHsp, 0.0);
    m_cachedBaseHessianSparsity = std::make_unique<SuiteSparseMatrix>(baseHsp);

    return baseHsp;
}

SuiteSparseMatrix InflatablePeriodicUnit::bentSheetHessianSparsityPattern(Real val) const {
    if (m_cachedBentSheetHessianSparsity) {
        if (m_cachedBentSheetHessianSparsity->Ax[0] != val) m_cachedBentSheetHessianSparsity->fill(val);
        return *m_cachedBentSheetHessianSparsity;
    }
    SuiteSparseMatrix baseHsp = baseHessianSparsityPattern();

    SuiteSparseMatrix Hsp(sheet.numVars() + numMacroRVars(), sheet.numVars() + numMacroRVars());
    Hsp.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    Hsp.Ap.reserve(sheet.numVars() + numMacroRVars() + 1);

    auto &Ap = Hsp.Ap;
    auto &Ai = Hsp.Ai;

    Ap.insert(Ap.begin(), baseHsp.Ap.begin(), baseHsp.Ap.end());
    Ai.insert(Ai.begin(), baseHsp.Ai.begin(), baseHsp.Ai.end());

    // Insert the sparsity for the kappa and alpha.
    for (size_t i = 0; i < numMacroRVars(); ++i) {
        for (size_t j = 0; j < sheet.numVars() + i + 1; ++j) {
            Ai.push_back(j);
        }
        Ap.push_back(Ai.size());
    }

    Hsp.nz = Ai.size();
    Hsp.Ax.assign(Hsp.nz, val);

    m_cachedBentSheetHessianSparsity = std::make_unique<SuiteSparseMatrix>(Hsp);

    return Hsp;
}

SuiteSparseMatrix InflatablePeriodicUnit::hessianSparsityPattern(Real val) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("InflatablePeriodicUnit.hessianSparsityPattern");
    if (m_cachedHessianSparsity) {
        if (m_cachedHessianSparsity->Ax[0] != val) m_cachedHessianSparsity->fill(val);
        return *m_cachedHessianSparsity;
    }

    TMatrix Hsp(numVars(), numVars());
    Hsp.symmetry_mode = TMatrix::SymmetryMode::UPPER_TRIANGLE;
    Hsp.pruneTol = -1.0;

    auto baseHsp = m_planar_homogenization ? baseHessianSparsityPattern() : bentSheetHessianSparsityPattern();
    // Need to fill the sparsity pattern with one so that we can compute A^T Hsp A.
    baseHsp.fill(1.0);

    const SuiteSparseMatrix &A = m_periodicPatchToInflatableSheetMapTranspose;
    computeAHA(Hsp, baseHsp, A);
    auto Hsp_CSC = SuiteSparseMatrix(Hsp);
    Hsp_CSC.fill(val);

    m_cachedHessianSparsity = std::make_unique<SuiteSparseMatrix>(Hsp_CSC);

    return Hsp_CSC;
}

// Debug.
SuiteSparseMatrix InflatablePeriodicUnit::baseHessian(EnergyType etype) const {
    SuiteSparseMatrix H;
    baseHessian(H, etype);
    return H;
}

void InflatablePeriodicUnit::baseHessian(SuiteSparseMatrix &baseH, EnergyType etype) const {
    baseH = baseHessianSparsityPattern();
    // The hessian for gravity is zero, so we can directly convert etype to the SheetEnergyType.
    SheetEnergyType sheet_etype = (etype == EnergyType::Full) ? SheetEnergyType::Full : ((etype == EnergyType::Elastic) ? SheetEnergyType::Elastic : SheetEnergyType::Pressure);
    sheet.hessian(baseH, sheet_etype);
    if (etype == EnergyType::Full || etype == EnergyType::Pressure)
        hessianPeriodicPressurePotential(baseH);
    baseH *= 1.0 / m_initial_area;
}


SuiteSparseMatrix InflatablePeriodicUnit::hessian(EnergyType etype, bool kappaOnly) const {
    SuiteSparseMatrix H = hessianSparsityPattern();
    hessian(H, etype, kappaOnly);
    return H;
}

// Need to construct the sparsity pattern through the hessian. 
void InflatablePeriodicUnit::hessian(SuiteSparseMatrix &H, EnergyType etype, bool kappaOnly) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("InflatablePeriodicUnit.hessian");
    if (H.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE) throw std::runtime_error("InflatablePeriodicUnit.hessian: H must be upper triangular");
    SuiteSparseMatrix baseH;
    if (m_planar_homogenization) {
        baseHessian(baseH, etype);
    } else {
        baseH = bent_sheet_hessian(etype, kappaOnly);
    }

    // Compute A^T H A by exploiting the structure of A.
    computeAHA(H, baseH, m_sparse_periodicPatchToInflatableSheetMapTranspose);

    baseH.reflectUpperTriangle();

    MX3d apply_dx_d_F;
    apply_dx_d_F.resize(sheet.numVars() + numMacroRVars(), 3);
    apply_dx_d_F.setZero();

    for (size_t i = 0; i < 3; ++i) baseH.applyTransposeParallel(m_d_x_d_F.col(i), apply_dx_d_F.col(i));

    MX3d d_uF_d_F;
    d_uF_d_F.resize(numMacraVars() + numFluctuationDisplacementVars(), 3);
    d_uF_d_F.setZero();

    for (size_t i = 0; i < 3; ++i) {
        m_sparse_periodicPatchToInflatableSheetMap.applyTransposeParallel(apply_dx_d_F.col(i), d_uF_d_F.col(i));
    }

    MX3d d_u_d_F;
    d_u_d_F.resize(numMacraVars() + numFluctuationDisplacementVars() - numMacroFVars(), 3);
    d_u_d_F.setZero();

    for (size_t i = 0; i < 3; ++i) {
        d_u_d_F.col(i) = d_uF_d_F.col(i).tail(numMacraVars() + numFluctuationDisplacementVars() - numMacroFVars());
    }

    M3d d_F_d_F = m_d_x_d_F.transpose() * apply_dx_d_F;

    H.addNZBlock(0, numMacroFVars(), d_u_d_F.transpose());
    H.addNZBlock(0, 0, d_F_d_F);
}

const x_hat_sensitivity &InflatablePeriodicUnit::SensitivityCache::lookup(size_t vi) const { return sensitivityForBending.at(vi); }

void InflatablePeriodicUnit::SensitivityCache::update(const InflatablePeriodicUnit &ipu) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("InflatablePeriodicUnit.sensitivitycache.update");
    if (!sensitivityForBending.empty()) return;
    sensitivityForBending.resize(ipu.sheet.numVars() / 3);
    VXd x_flat = ipu.get_x_flat();
    auto processVertex = [this, &ipu, &x_flat] (size_t vi) {
        sensitivityForBending[vi].update(ipu.get_alpha(), ipu.get_kappa(), ipu.get_grad_omega(), ipu.get_axis_mat(), x_flat.segment<3>(3 * vi) - ipu.get_center(), true);
    };
    parallel_for_range(ipu.sheet.numVars() / 3, processVertex);
}

// Out-of-line constructor and destructor needed because ConstrainedEdgeSensitivity<Real_> is an incomplete type upon declaration of sensitivityForConstrainedEdge.
InflatablePeriodicUnit::SensitivityCache:: SensitivityCache() { }
InflatablePeriodicUnit::SensitivityCache::~SensitivityCache() { }
void InflatablePeriodicUnit::SensitivityCache::clear() { sensitivityForBending.clear(); }

// The gradient of the energy over sheet variables configured by the bending variables but without periodic boundary conditions. 
InflatablePeriodicUnit::VXd InflatablePeriodicUnit::bent_sheet_gradient(EnergyType etype) const {
    {
        m_sensitivityCache.update(*this);
    }     
    VXd sheetGradient = getSheetGradient(etype);
    // First compute the gradient w.r.t. to x, kappa, a, i.e. the bending wrapping variables without periodic boundary conditions.
    VXd result(sheetGradient.size() + numMacroRVars());

    result.setZero();
    
    for (size_t i = 0; i < size_t(sheetGradient.size() / 3); ++i) {
        const auto &sensitivity = m_sensitivityCache.lookup(i);
        V3d dE_d_x_hat = sheetGradient.segment<3>(3 * i);
        VXd grad_result = sensitivity.jacobian.transpose() * dE_d_x_hat;
        // dE_d_x
        result.segment<3>(3 * i) += grad_result.segment<3>(0);
        // dE_d_kappa, dE_d_alpha
        result.segment<2>(sheetGradient.size()) += grad_result.segment<2>(3);
    }
    return result;
}


template<typename Real_>
struct dx_hat_dr_entry {
    typename CSCMatrix<SuiteSparse_long, Real_>::index_type first;
    typename CSCMatrix<SuiteSparse_long, Real_>::value_type second;
};
template<typename Real_>
using dx_hat_dr_type = std::vector<std::vector<dx_hat_dr_entry<Real_>>>;


// This function compute the same content as SuiteSparseMatrix result of `fill_dx_hat_d_x_d_x_hat_d_B` but stores the result in a different format.
template<typename Real_, typename SCPtr>
void get_dx_hat_dr(dx_hat_dr_type<Real_> &dx_hat_dr /* output */, SCPtr bending_sensitivity, size_t nv) {   
    BENCHMARK_SCOPED_TIMER_SECTION timer("InflatablePeriodicUnit.get_dx_hat_dr");

    using index_type = SuiteSparse_long;

    dx_hat_dr.resize(nv * 3);
    // m_kappa and m_alpha affect all vertices. 
    // x_i affects x_hat_i
    for (auto &row : dx_hat_dr) { row.clear(); row.reserve(5); } // At most 5 reduced variables affect each variable.

    for (size_t k = 0; k < nv; ++k) {
        const auto &jacobian = bending_sensitivity->lookup(k).jacobian;
        for (size_t ri = 0; ri < 3; ++ri) {
            for (size_t xi = 0; xi < 3; ++xi) {
                dx_hat_dr[3 * k + ri].push_back({index_type(3 * k + xi), jacobian(ri, xi)});
            }
        }
    }
}

template<typename SCPtr>
void fill_dx_hat_d_x_d_x_hat_d_B(SuiteSparseMatrix &dx_hat_d_x, Eigen::Matrix<Real, Eigen::Dynamic, 2> &dx_hat_d_B, SCPtr bending_sensitivity, size_t nv) {
    const SuiteSparse_long m = nv * 3, n = nv * 3;
    SuiteSparseMatrix result(m, n);
    result.nz = nv * 9;

    // Now we fill out the transpose of the map one column (arm segment) at a time:
    //         3 x_hat   [    ]
    //                  3 * x

    auto &Ai = result.Ai;
    auto &Ap = result.Ap;
    auto &Ax = result.Ax;

    Ai.reserve(result.nz);
    Ap.reserve(n + 1);

    Ap.push_back(0); // col 0 begin

    dx_hat_d_B.resize(nv * 3, 2);
    dx_hat_d_B.setZero();
    for (size_t k = 0; k < nv; ++k) {
        const auto &jacobian = bending_sensitivity->lookup(k).jacobian;
        for (size_t ri = 0; ri < 3; ++ri) {
            dx_hat_d_B(3 * k + ri, 0) = jacobian(ri, 3);
            dx_hat_d_B(3 * k + ri, 1) = jacobian(ri, 4);

            for (size_t xi = 0; xi < 3; ++xi) {
                Ai.push_back(3 * k + xi);
                Ax.push_back(jacobian(xi, ri));
            }
            Ap.push_back(Ai.size()); // col end
        }
    }
    
    if (Ai.size() != size_t(result.nz)) throw std::runtime_error("dx_hat_d_x: Size of Ai does not match result.nz");
    if (Ap.size() != size_t(n + 1)) throw std::runtime_error("dx_hat_d_x: Size of Ap does not match n + 1");
    dx_hat_d_x = std::move(result);
}

SuiteSparseMatrix InflatablePeriodicUnit::bent_sheet_hessian(EnergyType etype, bool kappaOnly) const {
    SuiteSparseMatrix H = bentSheetHessianSparsityPattern();
    bent_sheet_hessian(H, etype, kappaOnly);
    return H;
}

// The hessian of the energy over sheet variables configured by the bending variables but without periodic boundary conditions. 
void InflatablePeriodicUnit::bent_sheet_hessian(SuiteSparseMatrix &Hout, EnergyType etype, bool kappaOnly) const {        
    if (m_planar_homogenization) return;
    BENCHMARK_SCOPED_TIMER_SECTION timer("InflatablePeriodicUnit.bent_sheet_hessian");
    if (Hout.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE) throw std::runtime_error("bent_sheet_hessian must be upper triangular");
    SuiteSparseMatrix baseH;
    baseHessian(baseH, etype);
    m_sensitivityCache.update(*this);

    // Assemble A^T baseH A by exploiting the structure of A:
    //         dx_hat_d_x is block diagonal; sparse application is efficient
    //         dx_hat_d_B is dense; the best way to apply it is with the original hessian's applyTransposeParallel function.


    SuiteSparseMatrix dx_hat_d_x;
    MX2d dx_hat_d_B;
    fill_dx_hat_d_x_d_x_hat_d_B(dx_hat_d_x, dx_hat_d_B, &m_sensitivityCache, sheet.numVars() / 3);
    BENCHMARK_START_TIMER_SECTION("InflatablePeriodicUnit.bent_sheet_hessian.base_hessian_assembly");
    // First compute the new hessian for the sheet variables.
    if (!kappaOnly) {
        dx_hat_dr_type<Real> dx_har_dr;
        get_dx_hat_dr(dx_har_dr, &m_sensitivityCache, sheet.numVars() / 3);
        // Accumulate contribution of each (upper triangle) entry in baseH to the
        // full Hessian term:
        //      dx_hat_k_dri baseH_kl dx_hat_l_drj
        using Idx = typename SuiteSparseMatrix::index_type;
        Idx idx = 0, idx2 = 0;
        Idx ncol = baseH.n, colbegin = baseH.Ap[0];
        // Sparse matrix multiplication: A^T baseH A where A is the block diagonal and contains the sensitivity of d_x_hat_d_x.
        for (Idx l = 0; l < ncol; ++l) {
            const Idx colend = baseH.Ap[l + 1];
            for (auto entry = colbegin; entry < colend; ++entry) {
                const Idx k = baseH.Ai[entry];
                const auto v = baseH.Ax[entry];
                if (!(k <= l)) throw std::runtime_error("bent_sheet_hessian: baseH must be upper triangular");
                const auto &dvk_dr = dx_har_dr[k];
                const auto &dvl_dr = dx_har_dr[l];
                for (const auto &dvl_drj : dvl_dr) {
                    const Idx j = dvl_drj.first;
                    if (dvk_dr.size() == 0) continue;
                    const auto val = dvl_drj.second * v;
                    {
                        const Idx i = dvk_dr[0].first;
                        if (i > j) continue;
                        idx = Hout.template addNZAtLoc</* _knownGood = */ false>(i, j, val * dvk_dr[0].second, idx);
                    }
                    for (size_t ii = 1; ii < dvk_dr.size(); ++ii) {
                        const Idx i = dvk_dr[ii].first;
                        if (i > j) break;
                        idx = Hout.template addNZAtLoc</* _knownGood = */ true>(i, j, val * dvk_dr[ii].second, idx);
                    }
                }
                if (k != l) {
                    // Contribution from (l, k), if it falls in the upper triangle of H; capture all the missed entries from the previous loop due to the (i>j) check.
                    for (const auto &dvk_drj : dvk_dr) {
                        const Idx j = dvk_drj.first;
                        if (dvl_dr.size() == 0) continue;
                        const auto val = dvk_drj.second * v;
                        {
                            const Idx i = dvl_dr[0].first;
                            if (i > j) continue;
                            idx2 = Hout.template addNZAtLoc</* _knownGood = */ false>(i, j, val * dvl_dr[0].second, idx2);
                        }
                        for (size_t ii = 1; ii < dvl_dr.size(); ++ii) {
                            const Idx i = dvl_dr[ii].first;
                            if (i > j) break;
                            idx2 = Hout.template addNZAtLoc</* _knownGood = */ true>(i, j, val * dvl_dr[ii].second, idx2);
                        }
                    }
                }
            }
            colbegin = colend;
        }
    }
    // Handle the dense columns from the change of variable jacobians separately to speed up the assembly.
    baseH.reflectUpperTriangle();

    MX2d apply_dx_hat_d_B;
    apply_dx_hat_d_B.resize(sheet.numVars(), 2);
    apply_dx_hat_d_B.setZero();

    baseH.applyTransposeParallel(dx_hat_d_B.col(0), apply_dx_hat_d_B.col(0));
    baseH.applyTransposeParallel(dx_hat_d_B.col(1), apply_dx_hat_d_B.col(1));

    M2d d_B_d_B = dx_hat_d_B.transpose() * apply_dx_hat_d_B;
    Hout.addNZBlock(sheet.numVars(), sheet.numVars(), d_B_d_B);


    MX2d d_x_d_B;
    d_x_d_B.resize(sheet.numVars(), 2);
    d_x_d_B.setZero();

    dx_hat_d_x.applyTransposeParallel(apply_dx_hat_d_B.col(0), d_x_d_B.col(0));
    dx_hat_d_x.applyTransposeParallel(apply_dx_hat_d_B.col(1), d_x_d_B.col(1));
    Hout.addNZBlock(0, sheet.numVars(), d_x_d_B);

    BENCHMARK_STOP_TIMER_SECTION("InflatablePeriodicUnit.bent_sheet_hessian.base_hessian_assembly");

    BENCHMARK_START_TIMER_SECTION("InflatablePeriodicUnit.bent_sheet_hessian.sensitivity_assembly");
    static constexpr size_t JointJacobianRows = 3;
    static constexpr size_t JointJacobianCols = 5;

    const size_t kappa_offset = 3, alpha_offset = 4;

    // Accumulate contribution of the Hessian of x_hat wrt the bending transformation variables.
        //  dE/x_hat^j (d^2 x_hat^j / dbend_var_k dbend_var_l)
    const VXd sheet_gradient = getSheetGradient(etype);
    for (size_t j = 0; j < sheet.numVars() / 3; ++j) {
        const auto &sensitivity = m_sensitivityCache.lookup(j);
        V3d dE_dx_hat = sheet_gradient.segment<3>(3 * j);
        Eigen::Matrix<Real, JointJacobianCols, JointJacobianCols> contrib;
        contrib = dE_dx_hat[0]  * sensitivity.hessian[0]
               +  dE_dx_hat[1]  * sensitivity.hessian[1]
               +  dE_dx_hat[2]  * sensitivity.hessian[2];

        Hout.addNZBlock(3 * j, 3 * j, contrib.block(0, 0, 3, 3));
        Hout.addNZBlock(3 * j, sheet.numVars(), contrib.template block<3, 1>(0, kappa_offset));
        Hout.addNZBlock(3 * j, sheet.numVars() + 1, contrib.template block<3, 1>(0, alpha_offset));
        Hout.addNZBlock(sheet.numVars(), sheet.numVars(), contrib.template block<2, 2>(kappa_offset, kappa_offset));
    }
    BENCHMARK_STOP_TIMER_SECTION("InflatablePeriodicUnit.bent_sheet_hessian.sensitivity_assembly");
}

std::shared_ptr<InflatablePeriodicUnit::Mesh> InflatablePeriodicUnit::visualizationMesh(bool duplicateFusedTris) const { 
    // Visualize (copies of) the periodic unit.
    auto inflatable_mesh = sheet.visualizationMesh(duplicateFusedTris); 
    if (visualizationTilePower == 0) return inflatable_mesh;
    auto doublePeriodicUnit = [this](std::vector<V3d> input_vertices, std::vector<MeshIO::IOElement> input_elements, size_t axis = 0, int period_multiplier = 1) -> std::pair<std::vector<V3d>, std::vector<MeshIO::IOElement>> {
        std::vector<V3d> vertices = input_vertices;
        std::vector<MeshIO::IOElement> elements = input_elements;
        const size_t nv = vertices.size();

        const size_t nt = elements.size();
        std::vector<V3d> merged_vertices = vertices;
        merged_vertices.reserve(2 * nv);
        std::vector<MeshIO::IOElement> merged_elements = elements;
        merged_elements.reserve(2 * nt);
        for (size_t vi = 0; vi < nv; ++vi) {
            merged_vertices.emplace_back(vertices[vi]);
            Real shift = m_inputMeshDimension[axis] * period_multiplier;
            merged_vertices[vi + nv][axis] += m_F(axis) * shift;
            merged_vertices[vi + nv][1 - axis] += m_F(2) * shift;
        }
        for (const auto &tri : elements) {
            merged_elements.emplace_back(tri[0] + nv,
                                         tri[1] + nv,
                                         tri[2] + nv);
        }
        return std::make_pair(merged_vertices, merged_elements);
    };

    std::vector<V3d> curr_vertices;
    std::vector<MeshIO::IOElement> curr_elements;
    curr_vertices.reserve(inflatable_mesh->vertices().size());
    curr_elements.reserve(inflatable_mesh->elements().size());
    for (const auto &v : inflatable_mesh->vertices()) curr_vertices.push_back(v.node()->p);
    for (const auto &e : inflatable_mesh->elements()) {
            curr_elements.emplace_back(e.node(0).volumeNode().index(),
                                    e.node(1).volumeNode().index(),
                                    e.node(2).volumeNode().index());
    }
    size_t axis = 0;
    for (size_t pi = 0; pi < visualizationTilePower; ++pi) {
        auto data_pair = doublePeriodicUnit(curr_vertices, curr_elements, axis, pi / 2 + 1);
        curr_vertices = data_pair.first;
        curr_elements = data_pair.second;
        axis = 1 - axis;
    }
    
    return std::make_shared<Mesh>(curr_elements, curr_vertices, /* suppressNonmanifoldWarning = */ !duplicateFusedTris);
}
