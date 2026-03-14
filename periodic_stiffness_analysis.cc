#include "periodic_stiffness_analysis.hh"

Eigen::VectorXd getHessianColHead(const SuiteSparseMatrix &hessian, size_t colIdx, size_t length) {
    Eigen::VectorXd col(length);
    col.setZero();
    for (size_t i = (size_t)hessian.Ap[colIdx]; i < (size_t)hessian.Ap[colIdx + 1]; i++) {
        if ((size_t)hessian.Ai[i] < length)
            col(hessian.Ai[i]) = hessian.Ax[i];
    }
    return col;
}


template<class IPU>
Real getBendingStiffnessHelper(IPU &ipu, SuiteSparseMatrix &hessian, NewtonOptimizer &opt, size_t num_Fu, std::vector<size_t> fixedVars) {
    // The stiffness computation assumes the structure is in equilibrium with respect to all variables except the bending variables (kappa, alpha).
    // Construct the RHS vector for the linear system to solve.
    Eigen::VectorXd minus_d_E_d_Fu_d_kappa = -1 * getHessianColHead(hessian, num_Fu, ipu.numVars());
    // Zero out the rows corresponding to fixed variables.
    for (size_t i = 0; i < fixedVars.size(); ++i) {
        minus_d_E_d_Fu_d_kappa(fixedVars[i]) = 0;
    }
    // Get the diagonal entry corresponding to kappa.
    Real dE_d_kappa_d_kappa = hessian.Ax[hessian.findDiagEntry(num_Fu)];
    // Solve the linear system.
    Eigen::VectorXd d_Fu_star_d_kappa;
    d_Fu_star_d_kappa.resize(ipu.numVars());
    d_Fu_star_d_kappa.setZero();
    opt.newton_step(d_Fu_star_d_kappa, -1 * minus_d_E_d_Fu_d_kappa);

    // // Optionally transform the fluctuation displacement perturbation to remove the vertical offset. 
    // if (!ipu.average_z_reparametrization) {
    //     Eigen::VectorXd curr_vars = ipu.getVars();
    //     ipu.setVars(d_Fu_star_d_kappa);
    //     // subtract the vertical offset from all the z coordinates of the fluctuation displacement perturbation.
    //     Real average_z = ipu.get_average_z();
    //     std::cout<<"average_z: "<<average_z<<std::endl;
    //     for (size_t i = 0; i < ipu.numFluctuationDisplacementVars() / 3; ++i) {
    //         d_Fu_star_d_kappa(ipu.numMacroFVars() + i * 3 + 2) -= average_z;
    //     }
    //     ipu.setVars(curr_vars);
    // }
    return dE_d_kappa_d_kappa - minus_d_E_d_Fu_d_kappa.dot(d_Fu_star_d_kappa) * ipu.areaFactor();
}

// template<class IPU>
// Eigen::VectorXd getBendingStiffness(IPU &ipu, Eigen::VectorXd alphas, NewtonOptimizer &opt, Real hessianShift, std::vector<size_t> fixedVars) {
//     // The stiffness computation assumes the structure is in equilibrium with respect to all variables except the bending variables (kappa, alpha).
//     Real original_alpha = ipu.get_alpha();
//     Eigen::VectorXd curr_vars = ipu.getVars();
//     size_t num_Fu = ipu.numMacroFVars() + ipu.numFluctuationDisplacementVars();

//     Eigen::VectorXd bending_stiffness(alphas.size());

//     for (size_t i = 0; i < (size_t)alphas.size(); i++) {
//         curr_vars(ipu.numVars() - 1) = alphas(i);
//         ipu.setVars(curr_vars); 
//         // Use the area factor to scale the hessian for energy density defined on the size of the patch at equilibrium state. Essentially we are treating the equilibrium state as the rest state when computing stiffness. 
//         SuiteSparseMatrix h = ipu.hessian();
//         h.scale(1.0 / ipu.areaFactor());
//         opt.solver().factorizeNumericWithShift(h, hessianShift);

//         bending_stiffness(i) = getBendingStiffnessHelper(ipu, h, opt.solver(), num_Fu, fixedVars);
//     }
//     // Restore the original state.
//     curr_vars(ipu.numVars() - 1) = original_alpha;
//     ipu.setVars(curr_vars);
//     return bending_stiffness;
// }

template<class IPU>
std::pair<Eigen::VectorXd, Eigen::VectorXd> getBendingStiffnessUsingBases(IPU &ipu, Eigen::VectorXd alphas, NewtonOptimizer &opt, Real hessianShift, std::vector<size_t> fixedVars) {
    // The stiffness computation assumes the structure is in equilibrium with respect to all variables except the bending variables (kappa, alpha).
    Real original_alpha = ipu.get_alpha();
    Eigen::VectorXd curr_vars = ipu.getVars();
    size_t num_Fu = ipu.numMacroFVars() + ipu.numFluctuationDisplacementVars();

    opt.setFixedVars(fixedVars);

    Eigen::VectorXd bending_stiffness(alphas.size());

    Eigen::VectorXd test_stiffness;
    test_stiffness.resize(5);
    Eigen::MatrixXd A;
    A.resize(5, 5);
    auto get_coeffs = [&](Real input_alpha) -> Eigen::VectorXd {
        Real c = cos(input_alpha);
        Real s = sin(input_alpha);
        Real c2 = c * c;
        Real s2 = s * s;
        Real c3 = c2 * c;
        Real s3 = s2 * s;
        Eigen::VectorXd coeffs;
        coeffs.resize(5);
        coeffs <<  c2 * s2, c3 * s, c * s3, c2 * c2, s2 * s2;
        return coeffs;
    };

    // Implement the following loop using a try block; if it fails, then apply a shift to theta.
    Real shift = 0;
    bool success = false;
    size_t n_attempts = 0;

    opt.update_factorizations_shiftedHessian(hessianShift * ipu.areaFactor());
    while ((!success) && (n_attempts < 3)) {
        try {
            std::cout<<"Stiffness control point shift: "<<shift<<std::endl;
            for (size_t i = 0; i < 5; ++i) {
                Real theta = i * M_PI / 5 + shift;
                std::cout<<"theta: "<<theta / M_PI * 180<<std::endl;
                curr_vars(ipu.numVars() - 1) = theta;
                ipu.setVars(curr_vars); 
                // Use the area factor to scale the hessian for energy density defined on the size of the patch at equilibrium state. Essentially we are treating the equilibrium state as the rest state when computing stiffness. 
                SuiteSparseMatrix h = ipu.hessian();
                h.scale(1.0 / ipu.areaFactor());
                test_stiffness(i) = getBendingStiffnessHelper(ipu, h, opt, num_Fu, fixedVars);
                A.row(i) = get_coeffs(theta);
            }
            std::cout<<"succeed attempt!"<<std::endl;
            success = true;
        } catch (std::runtime_error &e) {
            std::cout<<"Caught exception: "<<e.what()<<std::endl;
            std::cout<<"Applying shift to theta."<<std::endl;
            shift += M_PI * 0.1;
            n_attempts++;
        }
    }

    Eigen::VectorXd stiffness_coeff = A.colPivHouseholderQr().solve(test_stiffness);
    for (size_t i = 0; i < (size_t)alphas.size(); i++) {
        bending_stiffness(i) = get_coeffs(alphas(i)).dot(stiffness_coeff);
    }

    // Restore the original state.
    curr_vars(ipu.numVars() - 1) = original_alpha;
    ipu.setVars(curr_vars);
    return std::make_pair(bending_stiffness, stiffness_coeff);
}

// Debug
template<class IPU>
std::pair<Eigen::VectorXd, Eigen::VectorXd> get_bending_equilibrium_sensitivity(IPU &ipu, Real alpha, NewtonOptimizer &opt, Real hessianShift, std::vector<size_t> fixedVars) {
    // The stiffness computation assumes the structure is in equilibrium with respect to all variables except the bending variables (kappa, alpha).
    Real original_alpha = ipu.get_alpha();
    Eigen::VectorXd curr_vars = ipu.getVars();
    size_t num_Fu = ipu.numMacroFVars() + ipu.numFluctuationDisplacementVars();
    opt.setFixedVars(fixedVars);

    SuiteSparseMatrix h = ipu.hessian();

    curr_vars(ipu.numVars() - 1) = alpha;
    ipu.setVars(curr_vars); 
    opt.update_factorizations_shiftedHessian(hessianShift * ipu.areaFactor());



    // Construct the RHS vector for the linear system to solve.
    Eigen::VectorXd minus_d_E_d_Fu_d_kappa = -1 * getHessianColHead(h, num_Fu, ipu.numVars());
    // Zero out the rows corresponding to fixed variables.
    for (size_t i = 0; i < fixedVars.size(); ++i) {
        minus_d_E_d_Fu_d_kappa(fixedVars[i]) = 0;
    }
    // Get the diagonal entry corresponding to kappa.
    Real dE_d_kappa_d_kappa = h.Ax[h.findDiagEntry(num_Fu)];

    // Solve the linear system.
    Eigen::VectorXd d_Fu_star_d_kappa;
    d_Fu_star_d_kappa.resize(ipu.numVars());
    d_Fu_star_d_kappa.setZero();

    opt.newton_step(d_Fu_star_d_kappa, -1 * minus_d_E_d_Fu_d_kappa);
    d_Fu_star_d_kappa *= ipu.areaFactor();


    d_Fu_star_d_kappa.tail(ipu.numMacroRVars()).setZero();
    for (size_t i = 0; i < fixedVars.size(); ++i) {
        d_Fu_star_d_kappa(fixedVars[i]) = 0;
    }

    d_Fu_star_d_kappa(ipu.numVars() - 2) = h.Ax[h.findDiagEntry(num_Fu)];

    // Restore the original state.
    curr_vars(ipu.numVars() - 1) = original_alpha;
    ipu.setVars(curr_vars);
    return std::make_pair(d_Fu_star_d_kappa, minus_d_E_d_Fu_d_kappa);
}


template<class IPU>
Eigen::Matrix3d getTangentElasticityTensorHelper(const IPU &ipu, SuiteSparseMatrix &hessian, CholmodFactorizer &solver, std::vector<size_t> fixedVars) {
    // fixedVars should include F, kappa, alpha, and the fixed variables in the macro variables.
    // Use the full hessian.
    hessian.reflectUpperTriangle();
    // The stiffness computation assumes the structure is in equilibrium with respect to all variables except the average deformation gradient variables (F).
    // Construct the RHS vector for the linear system to solve.
    std::vector<Eigen::VectorXd> minus_d_E_d_u_d_F;
    minus_d_E_d_u_d_F.resize(3);
    for (size_t i = 0; i < 3; ++i) minus_d_E_d_u_d_F[i] = -1 * getHessianColHead(hessian, i, ipu.numVars());

    // Zero out the rows corresponding to fixed variables.
    for (size_t i = 0; i < fixedVars.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) minus_d_E_d_u_d_F[j](fixedVars[i]) = 0;
    }

    // get the 3x3 block of the hessian.
    Eigen::Matrix3d dE_dF_dF;
    for (size_t i = 0; i < 3; ++i) dE_dF_dF.col(i) = getHessianColHead(hessian, i, 3);

    // Solve three linear system.
    std::vector<Eigen::VectorXd> d_u_star_d_F;
    d_u_star_d_F.resize(3);
    for (size_t i = 0; i < 3; ++i) {
        d_u_star_d_F[i].resize(ipu.numVars());
        d_u_star_d_F[i].setZero();
        solver.solve(minus_d_E_d_u_d_F[i], d_u_star_d_F[i]);
    }

    Eigen::Matrix3d result;
    result.setZero();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
            result(i, j) = result(j, i) = -minus_d_E_d_u_d_F[i].dot(d_u_star_d_F[j]) + dE_dF_dF(i, j);
        }
    }
    return result;
}

template<class IPU>
ElasticityTensor<Real, 2> getTangentElasticityTensor(const IPU &ipu, SuiteSparseMatrix &hessian, CholmodFactorizer &solver, std::vector<size_t> fixedVars) {
    // fixedVars should include F, kappa, alpha, and the fixed variables in the macro variables.
    Eigen::Matrix3d result = getTangentElasticityTensorHelper(ipu, hessian, solver, fixedVars);
    for (size_t i = 0; i < 3; ++i) {
        result(i, 2) *= 0.5;
        result(2, i) *= 0.5;
    }

    ElasticityTensor<Real, 2> result_tensor;
    result_tensor.setD(result);
    return result_tensor;
}

template<class IPU>
Eigen::VectorXd getStretchingStiffness(const IPU &ipu, Eigen::VectorXd betas, NewtonOptimizer &opt, Real hessianShift, std::vector<size_t> fixedVars) {
    // The betas are specifying the direction of the stress and not the bending alpha variables. 

    // The stiffness computation assumes the structure is in equilibrium with respect to all variables except the average deformation gradient variables (F).
    
    SuiteSparseMatrix h = ipu.hessian();
    h.scale(1.0 / ipu.areaFactor());
    opt.update_factorizations_shiftedHessian(hessianShift);



    ElasticityTensor<Real, 2> tangentElasticityTensor = getTangentElasticityTensor(ipu, h, opt.solver, fixedVars);
    // Compute the inverse of the tensor.
    ElasticityTensor<Real, 2> inverse_tangentElasticityTensor = tangentElasticityTensor.inverse();
    Eigen::VectorXd stretching_stiffness(betas.size());
    for (size_t i = 0; i < (size_t)betas.size(); i++) {
        Eigen::Vector2d direction = Eigen::Vector2d(std::cos(betas(i)), std::sin(betas(i)));
        Real stiffness =  1.0 / (inverse_tangentElasticityTensor.doubleContractRank1(direction)).doubleContractRank1(direction);
        stretching_stiffness(i) = stiffness;
    }
    return stretching_stiffness;
}

template<class IPU>
Eigen::Matrix3d debugTangentElasticityTensor(const IPU &ipu, Eigen::VectorXd betas, NewtonOptimizer &opt, Real hessianShift, std::vector<size_t> fixedVars) {
    SuiteSparseMatrix h = ipu.hessian();
    h.scale(1.0 / ipu.areaFactor());
    opt.update_factorizations_shiftedHessian(hessianShift);
    return getTangentElasticityTensorHelper(ipu, h, opt.solver, fixedVars);
}

// Explicit instantiation for the two supported IPU types.
#include "InflatablePeriodicUnit.hh"
#include "InflatableMidSurfacePeriodicUnit.hh"
// template Eigen::VectorXd getBendingStiffness<InflatablePeriodicUnit>(InflatablePeriodicUnit &, Eigen::VectorXd, NewtonOptimizer &, Real, std::vector<size_t>);
template std::pair<Eigen::VectorXd, Eigen::VectorXd> getBendingStiffnessUsingBases<InflatablePeriodicUnit>(InflatablePeriodicUnit &, Eigen::VectorXd, NewtonOptimizer &, Real, std::vector<size_t>);
template std::pair<Eigen::VectorXd, Eigen::VectorXd> get_bending_equilibrium_sensitivity<InflatablePeriodicUnit>(InflatablePeriodicUnit &, Real, NewtonOptimizer &, Real, std::vector<size_t>);
template Eigen::VectorXd getStretchingStiffness<InflatablePeriodicUnit>(const InflatablePeriodicUnit &, Eigen::VectorXd, NewtonOptimizer &, Real, std::vector<size_t>);
template Eigen::Matrix3d debugTangentElasticityTensor<InflatablePeriodicUnit>(const InflatablePeriodicUnit &, Eigen::VectorXd, NewtonOptimizer &, Real, std::vector<size_t>);




// template Eigen::VectorXd getBendingStiffness<InflatableMidSurfacePeriodicUnit>(InflatableMidSurfacePeriodicUnit &, Eigen::VectorXd, NewtonOptimizer &, Real, std::vector<size_t>);
template std::pair<Eigen::VectorXd, Eigen::VectorXd> getBendingStiffnessUsingBases<InflatableMidSurfacePeriodicUnit>(InflatableMidSurfacePeriodicUnit &, Eigen::VectorXd, NewtonOptimizer &, Real, std::vector<size_t>);
template std::pair<Eigen::VectorXd, Eigen::VectorXd> get_bending_equilibrium_sensitivity<InflatableMidSurfacePeriodicUnit>(InflatableMidSurfacePeriodicUnit &, Real, NewtonOptimizer &, Real, std::vector<size_t>);
