#ifndef BSISP_HH
#define BSISP_HH

#include <cmath>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Utilities/ArrayPadder.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include <memory>
#include <string>
#include <atomic>

struct BendingStiffnessIntegralSensitivityPattern {
#if INFLATABLES_LONG_DOUBLE
    using Real = long double;
#else
    using Real = double;
#endif
    using VXd = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using MXd = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

    Real objective = 0;
    
    static constexpr size_t num_delta_q = 6;
    static constexpr size_t num_q = 5;
    
    size_t num_psi_p = 3;

    // Derivative of bending regularization w.r.t. delta, q.
    VXd delta_q_gradient = VXd::Zero(num_delta_q);
    Eigen::Matrix<Real, num_delta_q, num_delta_q> delta_q_hessian;

    // Derivative of bending regularization w.r.t. psi, alpha, beta
    VXd psi_p_gradient = VXd::Zero(num_psi_p);
    MXd psi_p_hessian;

    /////////////////////
    void update_delta_q(const Real delta, Real k1, Real k2, const Eigen::VectorXd stiffness_coefficients, Eigen::VectorXd activation = Eigen::VectorXd::Ones(num_q)) {
        // The formulas below are generated using mathematica.
        
        // Term 1: 1/64 (2 (3 k1^2 + 2 k1 k2 + 3 k2^2) \[Pi] - (k1 - k2)^2 \[Pi] Cos[4 \[Delta]])
        Real term_1 = 1.0 / 64.0 * (2 * (3 * pow(k1, 2) + 2 * k1 * k2 + 3 * pow(k2, 2)) * M_PI - pow(k1 - k2, 2) * M_PI * cos(4 * delta));
        // Term 2: 1/32 (k1 - k2) \[Pi] (-4 (k1 + k2) + (k1 - k2) Cos[2 \[Delta]]) Sin[2 \[Delta]]
        Real term_2 = 1.0 / 32.0 * (k1 - k2) * M_PI * (-4 * (k1 + k2) + (k1 - k2) * cos(2 * delta)) * sin(2 * delta);
        // Term 3: -(1/32) (k1 - k2) \[Pi] (4 (k1 + k2) + (k1 - k2) Cos[2 \[Delta]]) Sin[2 \[Delta]]
        Real term_3 = -(1.0 / 32.0) * (k1 - k2) * M_PI * (4 * (k1 + k2) + (k1 - k2) * cos(2 * delta)) * sin(2 * delta);
        // Term 4: 1/64 \[Pi] (6 (3 k1^2 + 2 k1 k2 + 3 k2^2) + 16 (-k1^2 + k2^2) Cos[2 \[Delta]] + (k1 - k2)^2 Cos[4 \[Delta]])
        Real term_4 = 1.0 / 64.0 * M_PI * (6 * (3 * pow(k1, 2) + 2 * k1 * k2 + 3 * pow(k2, 2)) + 16 * (-pow(k1, 2) + pow(k2, 2)) * cos(2 * delta) + pow(k1 - k2, 2) * cos(4 * delta));
        // Term 5: 1/64 \[Pi] (6 (3 k1^2 + 2 k1 k2 + 3 k2^2) + 16 (k1 - k2) (k1 + k2) Cos[2 \[Delta]] + (k1 - k2)^2 Cos[4 \[Delta]])
        Real term_5 = 1.0 / 64.0 * M_PI * (6 * (3 * pow(k1, 2) + 2 * k1 * k2 + 3 * pow(k2, 2)) + 16 * (k1 - k2) * (k1 + k2) * cos(2 * delta) + pow(k1 - k2, 2) * cos(4 * delta));

        std::vector<Real> terms = {term_1, term_2, term_3, term_4, term_5};

        objective = 0;
        for (size_t i = 0; i < num_q; ++i) {
            if (activation(i) > 0) objective += stiffness_coefficients(i) * terms[i];
        }
        delta_q_gradient.setZero();
        for (size_t i = 0; i < num_q; ++i) {
            if (activation(i) > 0) delta_q_gradient(i + 1) = terms[i];
        }

        // term 1 gradient: 1/16 (k1 - k2)^2 \[Pi] Sin[4 \[Delta]]
        Real term_1_gradient = 1.0 / 16.0 * M_PI * pow(k1 - k2, 2) * sin(4 * delta);
        // term 2 gradient: 1/16 (k1 - k2) \[Pi] Cos[2 \[Delta]] (-4 (k1 + k2) + (k1 - k2) Cos[2 \[Delta]]) - 1/16 (k1 - k2)^2 \[Pi] Sin[2 \[Delta]]^2
        Real term_2_gradient = 1.0 / 16.0 * M_PI * (k1 - k2) * cos(2 * delta) * (-4 * (k1 + k2) + (k1 - k2) * cos(2 * delta)) - 1.0 / 16.0 * M_PI * pow(k1 - k2, 2) * pow(sin(2 * delta), 2);
        // term 3 gradient: -(1/16) (k1 - k2) \[Pi] Cos[2 \[Delta]] (4 (k1 + k2) + (k1 - k2) Cos[2 \[Delta]]) + 1/16 (k1 - k2)^2 \[Pi] Sin[2 \[Delta]]^2
        Real term_3_gradient = -(1.0 / 16.0) * (k1 - k2) * M_PI * cos(2 * delta) * (4 * (k1 + k2) + (k1 - k2) * cos(2 * delta)) + 1.0 / 16.0 * M_PI * pow(k1 - k2, 2) * pow(sin(2 * delta), 2);
        // term 4 gradient: 1/64 \[Pi] (-32 (-k1^2 + k2^2) Sin[2 \[Delta]] - 4 (k1 - k2)^2 Sin[4 \[Delta]])
        Real term_4_gradient = 1.0 / 64.0 * M_PI * ((-32 * (-pow(k1, 2) + pow(k2, 2)) * sin(2 * delta)) - (4 * pow(k1 - k2, 2) * sin(4 * delta)));
        // term 5 gradient: 1/64 \[Pi] (-32 (k1 - k2) (k1 + k2) Sin[2 \[Delta]] - 4 (k1 - k2)^2 Sin[4 \[Delta]])
        Real term_5_gradient = 1.0 / 64.0 * M_PI * ((-32 * (k1 - k2) * (k1 + k2) * sin(2 * delta)) - (4 * pow(k1 - k2, 2) * sin(4 * delta)));

        std::vector<Real> term_gradients = {term_1_gradient, term_2_gradient, term_3_gradient, term_4_gradient, term_5_gradient};
        
        for (size_t i = 0; i < num_q; ++i) {
            if (activation(i) > 0) delta_q_gradient(0) += stiffness_coefficients(i) * term_gradients[i];
        }

        // Only the first row and columns are non zero since the function linear w.r.t. q. 
        delta_q_hessian.setZero();
        // term 1 hessian: 1/4 (k1 - k2)^2 \[Pi] Cos[4 \[Delta]]
        Real term_1_hessian = 1.0 / 4.0 * M_PI * pow(k1 - k2, 2) * cos(4 * delta);
        // term 2 hessian: -(3/8) (k1 - k2)^2 \[Pi] Cos[2 \[Delta]] Sin[2 \[Delta]] - 1/8 (k1 - k2) \[Pi] (-4 (k1 + k2) + (k1 - k2) Cos[2 \[Delta]]) Sin[2 \[Delta]]
        Real term_2_hessian = -(3.0 / 8.0) * pow(k1 - k2, 2) * M_PI * cos(2 * delta) * sin(2 * delta) - 1.0 / 8.0 * (k1 - k2) * M_PI * (-4 * (k1 + k2) + (k1 - k2) * cos(2 * delta)) * sin(2 * delta);
        // term 3 hessian: 3/8 (k1 - k2)^2 \[Pi] Cos[2 \[Delta]] Sin[2 \[Delta]] + 1/8 (k1 - k2) \[Pi] (4 (k1 + k2) + (k1 - k2) Cos[2 \[Delta]]) Sin[2 \[Delta]]
        Real term_3_hessian = 3.0 / 8.0 * pow(k1 - k2, 2) * M_PI * cos(2 * delta) * sin(2 * delta) + 1.0 / 8.0 * (k1 - k2) * M_PI * (4 * (k1 + k2) + (k1 - k2) * cos(2 * delta)) * sin(2 * delta);
        // term 4 hessian: 1/64 \[Pi] (-64 (-k1^2 + k2^2) Cos[2 \[Delta]] - 16 (k1 - k2)^2 Cos[4 \[Delta]])
        Real term_4_hessian = 1.0 / 64.0 * M_PI * ((-64 * (-pow(k1, 2) + pow(k2, 2)) * cos(2 * delta)) - (16 * pow(k1 - k2, 2) * cos(4 * delta)));
        // term 5 hessian: 1/64 \[Pi] (-64 (k1 - k2) (k1 + k2) Cos[2 \[Delta]] - 16 (k1 - k2)^2 Cos[4 \[Delta]])
        Real term_5_hessian = 1.0 / 64.0 * M_PI * ((-64 * (k1 - k2) * (k1 + k2) * cos(2 * delta)) - (16 * pow(k1 - k2, 2) * cos(4 * delta)));

        std::vector<Real> term_hessians = {term_1_hessian, term_2_hessian, term_3_hessian, term_4_hessian, term_5_hessian};

        for (size_t i = 0; i < num_q; ++i) {
            if (activation(i) > 0) {
                delta_q_hessian(0, 0) += stiffness_coefficients(i) * term_hessians[i];

                delta_q_hessian(0, i + 1) = term_gradients[i];
                delta_q_hessian(i + 1, 0) = term_gradients[i];
            }
        }
    }

    void update(Real delta, Real k1, Real k2, const std::vector<Eigen::VectorXd> bsi_info) {
        if (bsi_info.size() != 1 + (num_psi_p - 1) + (num_psi_p - 1) * (num_psi_p - 1)) {
            throw std::runtime_error("Invalid number of bsi info!");
        }
        update_delta_q(delta, k1, k2, bsi_info[0]);
        // Jacobian: 
        //        delta    q
        // psi     -1      0
        // p1       0    partial p1
        // p2       0    partial p2
        // ...      0    ...
        MXd Jacobian = MXd::Zero(num_psi_p, num_delta_q);
        Jacobian(0, 0) = -1;
        for (size_t i = 0; i < num_q; ++i) {
            for (size_t j = 1; j < num_psi_p; ++j) {
                Jacobian(j, i + 1) = bsi_info[j](i);
            }
        }

        psi_p_gradient.resize(num_psi_p);
        psi_p_gradient.setZero();
        psi_p_gradient += Jacobian * delta_q_gradient;

        // Hessian:
        // delta: no hessian
        // q:
        //      psi   p1               p2             ...
        // psi   0     0                0
        // p1    0   partial p1       partial p1_p2   ...
        // p2    0   partial p1_p2    partial p2      ...
        // ...   0   ...              ...             ...
        std::vector<MXd> q_hessian;
        q_hessian.resize(num_q);
        for (size_t i = 0; i < num_q; ++i) {
            q_hessian[i] = MXd::Zero(num_psi_p, num_psi_p);
            for (size_t j = 0; j < num_psi_p - 1; ++j) {
                for (size_t k = 0; k < num_psi_p - 1; ++k) {
                    q_hessian[i](j + 1, k + 1) = bsi_info[1 + num_psi_p - 1 + j * (num_psi_p - 1) + k](i);
                }
            }
        }

        psi_p_hessian = MXd::Zero(num_psi_p, num_psi_p);
        psi_p_hessian += Jacobian * delta_q_hessian * Jacobian.transpose();

        // Add the hessian terms from the coefficients.
        for (size_t i = 0; i < num_q; ++i) {
            psi_p_hessian += delta_q_gradient(i + 1) * q_hessian[i];
        }
    }
};

#endif