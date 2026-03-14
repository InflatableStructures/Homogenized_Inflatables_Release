#ifndef PERIODIC_STIFFNESS_ANALYSIS_HH
#define PERIODIC_STIFFNESS_ANALYSIS_HH

#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include <memory>
#include <functional>

#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/SparseMatrices.hh>

////////////////////////////////////////////////////////////////////////////
// Bending Stiffness analysis
////////////////////////////////////////////////////////////////////////////

template<class IPU>
Real getBendingStiffnessHelper(IPU &ipu, SuiteSparseMatrix &hessian, CholmodFactorizer &solver, size_t num_Fu, std::vector<size_t> fixedVars = {});

template<class IPU>
Eigen::VectorXd getBendingStiffness(IPU &ipu, Eigen::VectorXd alphas, NewtonOptimizer &opt, Real hessianShift = 0, std::vector<size_t> fixedVars = {});

template<class IPU>
std::pair<Eigen::VectorXd, Eigen::VectorXd> getBendingStiffnessUsingBases(IPU &ipu, Eigen::VectorXd alphas, NewtonOptimizer &opt, Real hessianShift = 1e-10, std::vector<size_t> fixedVars = {});

template<class IPU>
std::pair<Eigen::VectorXd, Eigen::VectorXd> get_bending_equilibrium_sensitivity(IPU &ipu, Real alpha, NewtonOptimizer &opt, Real hessianShift = 0, std::vector<size_t> fixedVars = {});


////////////////////////////////////////////////////////////////////////////
// Stretching Stiffness analysis
////////////////////////////////////////////////////////////////////////////
template<class IPU>
Eigen::Matrix3d getTangentElasticityTensorHelper(const IPU &ipu, SuiteSparseMatrix &hessian, CholmodFactorizer &solver, std::vector<size_t> fixedVars = {});

template<class IPU>
ElasticityTensor<Real, 2> getTangentElasticityTensor(const IPU &ipu, SuiteSparseMatrix &hessian, CholmodFactorizer &solver, std::vector<size_t> fixedVars = {});

template<class IPU>
Eigen::VectorXd getStretchingStiffness(const IPU &ipu, Eigen::VectorXd betas, NewtonOptimizer &opt, Real hessianShift = 1e-10, std::vector<size_t> fixedVars = {});

template<class IPU>
Eigen::VectorXd get_stretching_equilibrium_sensitivity(const IPU &ipu, Real alpha, NewtonOptimizer &opt, Real hessianShift = 0, std::vector<size_t> fixedVars = {});

template<class IPU>
Eigen::Matrix3d debugTangentElasticityTensor(const IPU &ipu, Eigen::VectorXd betas, NewtonOptimizer &opt, Real hessianShift = 0, std::vector<size_t> fixedVars = {});


// Debug
// template<class IPU>
// Eigen::Matrix3d debugTangentElasticityTensor(const IPU &ipu, NewtonOptimizer &opt, Real hessianShift, std::vector<size_t> fixedVars);

#endif /* end of include guard: PERIODIC_STIFFNESS_ANALYSIS_HH */