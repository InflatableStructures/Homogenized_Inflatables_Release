#include "periodic_inflation_newton.hh"
#include <memory>

template<class ISheet>
struct PeriodicInflationNewtonProblem : public NewtonProblem {
    PeriodicInflationNewtonProblem(ISheet &isheet)
        : m_sheet(isheet), m_hessianSparsity(isheet.hessianSparsityPattern()) { }

    virtual void setVars(const Eigen::VectorXd &vars) override { m_sheet.setVars(vars.cast<typename ISheet::Real>()); }
    virtual const Eigen::VectorXd getVars() const override { return m_sheet.getVars().template cast<double>(); }
    virtual size_t numVars() const override { return m_sheet.numVars(); }

    virtual Real energy() const override { 
        Real result = m_sheet.systemEnergy();
        if (result > std::max(systemEnergyIncreaseFactorLimit * m_currSystemEnergy, energyLimitingThreshold))
             return safe_numeric_limits<Real>::max();
        return m_sheet.energy();
    }

    virtual Eigen::VectorXd gradient(bool /* freshIterate */ = false) const override {
        auto result = m_sheet.gradient();
        return result.template cast<double>();
    }

    void setCustomIterationCallback(const CallbackFunction &cb) { m_customCallback = cb; }


    virtual SuiteSparseMatrix hessianSparsityPattern() const override { /* m_hessianSparsity.fill(1.0); */ return m_hessianSparsity; }
    // The maximum factor by which we allow the elastic energy to increase in a single
    // Newton iteration; limiting this prevents large external forces from
    // severly deforming the mesh into a bad configuration.
    Real systemEnergyIncreaseFactorLimit = safe_numeric_limits<Real>::max();
    Real energyLimitingThreshold = 1e-6;

protected:
    virtual void m_evalHessian(SuiteSparseMatrix &result, bool /* projectionMask */) const override {
        result.setZero();
        m_sheet.hessian(result);

        if (hessianShift != 0.0) {
            result.addScaledIdentity(hessianShift);
        }
    }
    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        // TODO: mass matrix?
        result.setIdentity(true);
    }

    virtual void m_iterationCallback(size_t i) override { 
        m_currSystemEnergy = m_sheet.systemEnergy();
        if (m_customCallback) m_customCallback(i); 
        // return false; // don't exit early
    }

    CallbackFunction m_customCallback;

    ISheet &m_sheet;
    mutable SuiteSparseMatrix m_hessianSparsity;
    Real m_currSystemEnergy = safe_numeric_limits<Real>::max();

};

template<class ISheet>
std::unique_ptr<NewtonOptimizer> get_periodic_inflation_optimizer(ISheet &isheet, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, CallbackFunction customCallback, Real hessianShift, Real systemEnergyIncreaseFactorLimit, Real energyLimitingThreshold) {
    auto problem = std::make_unique<PeriodicInflationNewtonProblem<ISheet>>(isheet);
    problem->addFixedVariables(fixedVars);
    problem->setCustomIterationCallback(customCallback);
    problem->hessianShift = hessianShift;
    problem->systemEnergyIncreaseFactorLimit = systemEnergyIncreaseFactorLimit;
    problem->energyLimitingThreshold = energyLimitingThreshold;
    auto opt = std::make_unique<NewtonOptimizer>(std::move(problem));
    opt->options = opts;
    return opt;
}

template<class ISheet>
ConvergenceReport periodic_inflation_newton(ISheet &isheet, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, CallbackFunction customCallback, Real hessianShift, Real systemEnergyIncreaseFactorLimit, Real energyLimitingThreshold) {
    return get_periodic_inflation_optimizer(isheet, fixedVars, opts, customCallback, hessianShift, systemEnergyIncreaseFactorLimit, energyLimitingThreshold)->optimize();
}

// Explicit function template instantiations
#include "InflatablePeriodicUnit.hh"
#include "InflatableMidSurfacePeriodicUnit.hh"
template ConvergenceReport periodic_inflation_newton<InflatablePeriodicUnit  >(InflatablePeriodicUnit &,   const std::vector<size_t> &, const NewtonOptimizerOptions &, CallbackFunction, Real, Real, Real);
template ConvergenceReport periodic_inflation_newton<InflatableMidSurfacePeriodicUnit  >(InflatableMidSurfacePeriodicUnit &,   const std::vector<size_t> &, const NewtonOptimizerOptions &, CallbackFunction, Real, Real, Real);

// The following shouldn't be necessary, but fix an undefined symbol error when loading the `inflation` Python module
template std::unique_ptr<NewtonOptimizer> get_periodic_inflation_optimizer<InflatablePeriodicUnit  >(InflatablePeriodicUnit   &, const std::vector<size_t> &, const NewtonOptimizerOptions &, CallbackFunction, Real, Real, Real);
template std::unique_ptr<NewtonOptimizer> get_periodic_inflation_optimizer<InflatableMidSurfacePeriodicUnit  >(InflatableMidSurfacePeriodicUnit   &, const std::vector<size_t> &, const NewtonOptimizerOptions &, CallbackFunction, Real, Real, Real);