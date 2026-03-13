#ifndef BSIS_HH
#define BSIS_HH

#include <cmath>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Utilities/ArrayPadder.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include <memory>
#include <string>
#include <atomic>

struct BendingStiffnessIntegralSensitivity {
#if INFLATABLES_LONG_DOUBLE
    using Real = long double;
#else
    using Real = double;
#endif
    using  VXd = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    Real objective = 0;
    
    static constexpr size_t num_delta_q = 6;
    static constexpr size_t num_psi_a_b = 3;
    static constexpr size_t num_q = 5;


    // Derivative of bending regularization w.r.t. delta, q.
    VXd delta_q_gradient = VXd::Zero(num_delta_q);
    Eigen::Matrix<Real, num_delta_q, num_delta_q> delta_q_hessian;

    // Derivative of bending regularization w.r.t. psi, alpha, beta
    VXd psi_a_b_gradient = VXd::Zero(num_psi_a_b);
    Eigen::Matrix<Real, num_psi_a_b, num_psi_a_b> psi_a_b_hessian;

    /////////////////////
    void update_delta_q(Real delta, Real k1, Real k2, Eigen::VectorXd stiffness_coefficients, Eigen::VectorXd activation = Eigen::VectorXd::Ones(num_q)) {
        // The formulas below are generated using mathematica.
        // For i in range(5), Integrate[(k1 Cos[\[Theta]]^2 + k2 Sin[\[Theta]]^2)^2 * Cos[\[Delta] + \[Theta] + Pi / 2]^i *Sin[\[Delta] + \[Theta] + Pi / 2 ]^(4 - i), {\[Theta], 0, 2 Pi}]. See the mathematica notebook for more details.
        
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

    void update(Real delta, Real k1, Real k2, std::vector<Eigen::VectorXd> bsi_info) {
        Eigen::VectorXd stiffness_coefficients = bsi_info[0];
        Eigen::VectorXd coefficient_gradient_alpha = bsi_info[1];
        Eigen::VectorXd coefficient_gradient_beta = bsi_info[2];
        Eigen::VectorXd coefficient_hessian_alpha = bsi_info[3];
        Eigen::VectorXd coefficient_hessian_beta = bsi_info[4];
        Eigen::VectorXd coefficient_hessian_alpha_beta = bsi_info[5];

        update_delta_q(delta, k1, k2, stiffness_coefficients);
        // Jacobian: 
        //           delta    q
        // psi        -1      0
        // alpha       0    partial alpha
        // beta        0    partial beta
        Eigen::Matrix<Real, num_psi_a_b, num_delta_q> Jacobian;
        Jacobian.setZero();
        Jacobian(0, 0) = -1;
        for (size_t i = 0; i < num_q; ++i) {
            Jacobian(1, i + 1) = coefficient_gradient_alpha(i);
            Jacobian(2, i + 1) = coefficient_gradient_beta(i);
        }

        psi_a_b_gradient.setZero();
        psi_a_b_gradient += Jacobian * delta_q_gradient;

        // Hessian:
        // delta: no hessian
        // q:
        //      psi   alpha                   beta
        // psi   0      0                      0
        // alpha 0   partial alpha       partial alpha_beta
        // beta  0   partial alpha_beta  partial beta
        std::vector<Eigen::Matrix<Real, num_psi_a_b, num_psi_a_b>> q_hessian;
        q_hessian.resize(num_q);
        for (size_t i = 0; i < num_q; ++i) {
            q_hessian[i].setZero();
            q_hessian[i](1, 1) = coefficient_hessian_alpha(i);
            q_hessian[i](1, 2) = coefficient_hessian_alpha_beta(i);
            q_hessian[i](2, 1) = coefficient_hessian_alpha_beta(i);
            q_hessian[i](2, 2) = coefficient_hessian_beta(i);
        }

        psi_a_b_hessian.setZero();
        psi_a_b_hessian += Jacobian * delta_q_hessian * Jacobian.transpose();

        // Add the hessian terms from the coefficients.
        for (size_t i = 0; i < num_q; ++i) {
            psi_a_b_hessian += delta_q_gradient(i + 1) * q_hessian[i];
        }
    }
};

#endif