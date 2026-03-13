#include "InflatableMidSurfacePeriodicUnit.hh"
#include "MeshFEM/GlobalBenchmark.hh"


InflatableMidSurfacePeriodicUnit::InflatableMidSurfacePeriodicUnit(const std::shared_ptr<Mesh> &inMesh, const std::vector<bool> &fusedVtx, Real epsilon) 
    : ipu(inMesh, fusedVtx, epsilon) 
    {
            m_constructMidSurfaceToPeriodicPatchMapTranspose_all_vars();
    }

// void InflatableMidSurfacePeriodicUnit::m_constructMidSurfaceToPeriodicPatchMapTranspose() {
//     const SuiteSparse_long m = ipu.numFluctuationDisplacementVars() / 3 - 1, n = ipu.numFluctuationDisplacementVars() / 3 ;
//     SuiteSparseMatrix result(m, n);
//     result.nz = 2 * m;
//     // Now we fill out the transpose of the map one column (arm segment) at a time:
//     // # Average u_z vars [                ]
//     //                      # All u_z vars

//     auto &Ai = result.Ai;
//     auto &Ap = result.Ap;
//     auto &Ax = result.Ax;


//     Ai.reserve(result.nz);
//     Ap.reserve(n + 1);

//     Ap.push_back(0); // col 0 begin
//     for (size_t ai = 0; ai < (size_t)n; ++ai) {
//         if (ai > 0) {
//             Ai.push_back(ai-1);
//             Ax.push_back(-1);
//         }
//         if (ai < (size_t)m) {
//             Ai.push_back(ai);
//             Ax.push_back(1);
//         }
//         Ap.push_back(Ai.size()); // col end.
//     }

//     assert(Ai.size() == size_t(result.nz));
//     assert(Ap.size() == size_t(n+1      ));
//     m_midSurfaceToPeriodicPatchMapTranspose = std::move(result);
// }

void InflatableMidSurfacePeriodicUnit::m_constructMidSurfaceToPeriodicPatchMapTranspose_all_vars() {
    const SuiteSparse_long m = ipu.numVars();
    const SuiteSparse_long n = ipu.numVars();
    SuiteSparseMatrix result(m, n);
    // First we count the first nz - 1 new variables: each row has two entries.
    // Then we count the average z variable: this row has nz entries .
    // Then the macro variables and the x, y fluctuation variables each row has one entry.
    size_t nv = ipu.numFluctuationDisplacementVars() / 3;
    result.nz = ipu.numVars() + 2 * (nv - 1);
    // Now we fill out the tarnspose of the map one column (each joint angle) at a time:
    // # Average z Vars [            ]
    //                   # All z vars

    auto &Ax = result.Ax;
    auto &Ai = result.Ai;
    auto &Ap = result.Ap;

    Ax.reserve(result.nz);
    Ai.reserve(result.nz);
    Ap.push_back(0);
    size_t z_vx_count = 0;

    // Compute the multiplicity of periodic vertices:
    m_ipu_vx_multiplicity = VXd::Zero(ipu.numFluctuationDisplacementVars() / 3);
    for (size_t si = 0; si < ipu.sheet.numVars() / 3; ++si) {
        m_ipu_vx_multiplicity(ipu.get_IPU_vidx_for_inflatable_vidx(si)) += 1;
    }

    size_t az_var_idx = ipu.numMacroFVars() + ipu.numFluctuationDisplacementVars() - 1;
    for (size_t ai = 0; ai < (size_t)n; ++ai) {
        if ((ai < ipu.numMacroFVars()) or (ai >= ipu.numMacroFVars() + ipu.numFluctuationDisplacementVars())) {
            // This is a macroscopic variable
            Ai.push_back(ai);
            Ax.push_back(1);
        } else if ( ai < ipu.numVars() - ipu.numMacroRVars()) {
            // This is a fluctuation displacement variable
            if (ai % 3 == 2) {
                Real factor = 1.0 / m_ipu_vx_multiplicity(z_vx_count);
                // This is a z displacement variable
                // The -1 entry
                if (z_vx_count > 0) {
                    Ai.push_back(ai - 3);
                    Ax.push_back(-1 * factor);
                }
                // the +1 entry
                if (z_vx_count < nv - 1) {
                    Ai.push_back(ai);
                    Ax.push_back(1 * factor);
                }
                // the all 1 row for the az variable
                Ai.push_back(az_var_idx);
                Ax.push_back(1 * factor);
                z_vx_count += 1;
            } else {
                // This is a x or y displacement variable
                Ai.push_back(ai);
                Ax.push_back(1);
            }
        }
        Ap.push_back(Ai.size()); // col end.
    }

    assert(Ai.size() == size_t(result.nz));
    assert(Ap.size() == size_t(n+1      ));
    m_midSurfaceToPeriodicPatchMapTranspose_all_vars = std::move(result);
}


SuiteSparseMatrix InflatableMidSurfacePeriodicUnit::baseHessianSparsityPattern() const {
    BENCHMARK_SCOPED_TIMER_SECTION("InflatableMidSurfacePeriodicUnit::baseHessianSparsityPattern");
    if (m_cachedBaseHessianSparsity) return *m_cachedBaseHessianSparsity;
    auto baseHsp = ipu.hessianSparsityPattern();
    m_cachedBaseHessianSparsity = std::make_unique<SuiteSparseMatrix>(baseHsp);

    return baseHsp;
}

SuiteSparseMatrix InflatableMidSurfacePeriodicUnit::hessianSparsityPattern(Real val) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("InflatableMidSurfacePeriodicUnit.hessianSparsityPattern");
    if (m_cachedHessianSparsity) {
        if (m_cachedHessianSparsity->Ax[0] != val) m_cachedHessianSparsity->fill(val);
        return *m_cachedHessianSparsity;
    }

    TMatrix Hsp(numVars(), numVars());
    Hsp.symmetry_mode = TMatrix::SymmetryMode::UPPER_TRIANGLE;
    Hsp.pruneTol = -1.0;

    auto baseHsp = baseHessianSparsityPattern();
    // Need to fill the sparsity pattern with one so that we can compute A^T Hsp A.
    baseHsp.fill(1.0);

    const SuiteSparseMatrix &A = m_midSurfaceToPeriodicPatchMapTranspose_all_vars;
    ipu.computeAHA(Hsp, baseHsp, A);
    auto Hsp_CSC = SuiteSparseMatrix(Hsp);
    Hsp_CSC.fill(val);

    m_cachedHessianSparsity = std::make_unique<SuiteSparseMatrix>(Hsp_CSC);

    return Hsp_CSC;
}

SuiteSparseMatrix InflatableMidSurfacePeriodicUnit::hessian(EnergyType etype, bool kappaOnly) const {
    SuiteSparseMatrix H = hessianSparsityPattern();
    hessian(H, etype, kappaOnly);
    return H;
}

// Need to construct the sparsity pattern through the hessian. 
void InflatableMidSurfacePeriodicUnit::hessian(SuiteSparseMatrix &H, EnergyType etype, bool kappaOnly) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("InflatableMidSurfacePeriodicUnit.hessian");
    assert(H.symmetry_mode == SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE);
    SuiteSparseMatrix baseH;
    baseH = baseHessianSparsityPattern();
    ipu.hessian(baseH, etype, kappaOnly);


    // Compute A^T H A by exploiting the structure of A.
    ipu.computeAHA(H, baseH, m_midSurfaceToPeriodicPatchMapTranspose_all_vars);
}
