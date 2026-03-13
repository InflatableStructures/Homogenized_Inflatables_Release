#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <InflatablePeriodicUnit.hh>
#include <InflatableMidSurfacePeriodicUnit.hh>
#include <periodic_stiffness_analysis.hh>
#include <BendingChangeVarSensitivity.hh>

#include "periodic_inflation_newton.cc"


#include <MeshFEM/../../python_bindings/MeshEntities.hh>
#include <MeshFEM/../../python_bindings/BindingUtils.hh>

#if INFLATABLES_LONG_DOUBLE
#include "extended_precision.hh"
#endif

PYBIND11_MODULE(homogenized_inflation, m) {
    m.doc() = "Homogenized Inflation simulation";
    py::module detail_module = m.def_submodule("detail");

    py::module::import("MeshFEM");
    py::module::import("mesh");

    py::module::import("Inflation");
    py::module::import("py_newton_optimizer");

    ////////////////////////////////////////////////////////////////////////////////
    // Mesh construction (for mesh type used by inflation routines)
    ////////////////////////////////////////////////////////////////////////////////
    using Mesh = InflatableSheet::Mesh;
    // WARNING: Mesh's holder type is a shared_ptr; returning a unique_ptr will lead to a dangling pointer in the current version of Pybind11
    m.def("Mesh", [](const std::string &path) { return std::shared_ptr<Mesh>(Mesh::load(path)); }, py::arg("path"));
    m.def("Mesh", [](const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F) { return std::make_shared<Mesh>(F, V); }, py::arg("V"), py::arg("F"));

    ////////////////////////////////////////////////////////////////////////////////
    // Free-standing functions
    ////////////////////////////////////////////////////////////////////////////////
    m.def("inflation_newton", &periodic_inflation_newton<  InflatablePeriodicUnit>, py::arg("ipu"),                  py::arg("fixedVars"), py::arg("options") = NewtonOptimizerOptions(), py::arg("callback") = nullptr, py::arg("hessianShift") = 0.0, py::arg("systemEnergyIncreaseFactorLimit") = safe_numeric_limits<Real>::max(), py::arg("energyLimitingThreshold") = 1e-6);
    m.def("inflation_newton", &periodic_inflation_newton<  InflatableMidSurfacePeriodicUnit>, py::arg("ipu"),                  py::arg("fixedVars"), py::arg("options") = NewtonOptimizerOptions(), py::arg("callback") = nullptr, py::arg("hessianShift") = 0.0, py::arg("systemEnergyIncreaseFactorLimit") = safe_numeric_limits<Real>::max(), py::arg("energyLimitingThreshold") = 1e-6);
    m.def("get_inflation_optimizer", &get_periodic_inflation_optimizer<  InflatablePeriodicUnit>, py::arg("ipu"),                  py::arg("fixedVars"), py::arg("options") = NewtonOptimizerOptions(), py::arg("callback") = nullptr, py::arg("hessianShift") = 0.0, py::arg("systemEnergyIncreaseFactorLimit") = safe_numeric_limits<Real>::max(), py::arg("energyLimitingThreshold") = 1e-6);
    m.def("get_inflation_optimizer", &get_periodic_inflation_optimizer<  InflatableMidSurfacePeriodicUnit>, py::arg("ipu"),                  py::arg("fixedVars"), py::arg("options") = NewtonOptimizerOptions(), py::arg("callback") = nullptr, py::arg("hessianShift") = 0.0, py::arg("systemEnergyIncreaseFactorLimit") = safe_numeric_limits<Real>::max(), py::arg("energyLimitingThreshold") = 1e-6);

    // Stiffness analysis
    // m.def("getBendingStiffness", &getBendingStiffness<InflatablePeriodicUnit>, py::arg("ipu"), py::arg("alphas"), py::arg("optimizer"), py::arg("hessianShift"), py::arg("useFixedVars"));
    // m.def("getBendingStiffness", &getBendingStiffness<InflatableMidSurfacePeriodicUnit>, py::arg("mid_ipu"), py::arg("alphas"), py::arg("optimizer"), py::arg("hessianShift"), py::arg("fixedVars"));

    m.def("getBendingStiffnessUsingBases", &getBendingStiffnessUsingBases<InflatablePeriodicUnit>, py::arg("ipu"), py::arg("alphas"), py::arg("optimizer"), py::arg("hessianShift"), py::arg("useFixedVars"));
    m.def("getBendingStiffnessUsingBases", &getBendingStiffnessUsingBases<InflatableMidSurfacePeriodicUnit>, py::arg("mid_ipu"), py::arg("alphas"), py::arg("optimizer"), py::arg("hessianShift"), py::arg("fixedVars"));

    m.def("get_bending_equilibrium_sensitivity", &get_bending_equilibrium_sensitivity<InflatablePeriodicUnit>, py::arg("ipu"), py::arg("alphas"), py::arg("optimizer"), py::arg("hessianShift"), py::arg("useFixedVars"));
    m.def("get_bending_equilibrium_sensitivity", &get_bending_equilibrium_sensitivity<InflatableMidSurfacePeriodicUnit>, py::arg("mid_ipu"), py::arg("alphas"), py::arg("optimizer"), py::arg("hessianShift"), py::arg("fixedVars"));

    m.def("getStretchingStiffness", &getStretchingStiffness<InflatablePeriodicUnit>, py::arg("ipu"), py::arg("betas"), py::arg("optimizer"), py::arg("hessianShift"), py::arg("fixedVars"));
    m.def("debugTangentElasticityTensor", &debugTangentElasticityTensor<InflatablePeriodicUnit>, py::arg("ipu"), py::arg("betas"), py::arg("optimizer"), py::arg("hessianShift"), py::arg("fixedVars"));

    ////////////////////////////////////////////////////////////////////////////////
    // Inflatable Periodic Unit
    ////////////////////////////////////////////////////////////////////////////////

    py::class_<InflatablePeriodicUnit, std::shared_ptr<InflatablePeriodicUnit>> pyInflatablePeriodicUnit(m, "InflatablePeriodicUnit");

    using IPUEType = InflatablePeriodicUnit::EnergyType;
    py::enum_<InflatablePeriodicUnit::EnergyType>(pyInflatablePeriodicUnit, "EnergyType")
        .value("Full"    , IPUEType::Full)
        .value("Elastic" , IPUEType::Elastic)
        .value("Pressure", IPUEType::Pressure)
        ;

    pyInflatablePeriodicUnit
        .def(py::init<const std::shared_ptr<Mesh> &, const std::vector<bool> &, Real>(), py::arg("mesh"), py::arg("fusedVtx") = std::vector<bool>(), py::arg("epsilon") = 1e-7)

        .def("numMacroFVars", [](const InflatablePeriodicUnit &ipu) { return InflatablePeriodicUnit::numMacroFVars(); })
        .def("numFluctuationDisplacementVars", &InflatablePeriodicUnit::numFluctuationDisplacementVars)
        .def("numVars",      &InflatablePeriodicUnit::numVars)
        .def("getVars",      &InflatablePeriodicUnit::getVars)
        .def("setVars",      &InflatablePeriodicUnit::setVars)
        .def("reparametrize_vertical_offset", &InflatablePeriodicUnit::reparametrize_vertical_offset)
        .def("energy",   &InflatablePeriodicUnit::energy  , py::arg("energyType") = IPUEType::Full)
        .def("gradient", &InflatablePeriodicUnit::gradient, py::arg("energyType") = IPUEType::Full)

        .def("bent_sheet_getVars",      &InflatablePeriodicUnit::bent_sheet_getVars)
        .def("bent_sheet_setVars",      &InflatablePeriodicUnit::bent_sheet_setVars)
        .def("bent_sheet_gradient", &InflatablePeriodicUnit::bent_sheet_gradient, py::arg("energyType") = IPUEType::Full)
        .def("bent_sheet_numVars",      &InflatablePeriodicUnit::bent_sheet_numVars)

        .def("hessianSparsityPattern", &InflatablePeriodicUnit::hessianSparsityPattern, py::arg("val") = 1.0)
        .def("hessian",  py::overload_cast<IPUEType, bool>(&InflatablePeriodicUnit::hessian, py::const_), py::arg("energyType") = IPUEType::Full, py::arg("kappaOnly") = false)
        .def("bent_sheet_hessian",  py::overload_cast<IPUEType, bool>(&InflatablePeriodicUnit::bent_sheet_hessian, py::const_), py::arg("energyType") = IPUEType::Full, py::arg("kappaOnly") = false)
        .def("base_hessian",  py::overload_cast<IPUEType>(&InflatablePeriodicUnit::baseHessian, py::const_), py::arg("energyType") = IPUEType::Full)

        .def_readonly("sheet", &InflatablePeriodicUnit::sheet, py::return_value_policy::reference)
        .def("getPeriodicPatchToInflatableSheetMapTranspose", &InflatablePeriodicUnit::getPeriodicPatchToInflatableSheetMapTranspose, py::return_value_policy::reference)
        .def("get_IPU_vidx_for_inflatable_vidx", &InflatablePeriodicUnit::get_IPU_vidx_for_inflatable_vidx)
        .def("get_use_planar_homogenization", &InflatablePeriodicUnit::get_use_planar_homogenization)
        .def("get_deformation_scale_factors", &InflatablePeriodicUnit::get_deformation_scale_factors)
        .def("get_center_fixedVars", &InflatablePeriodicUnit::get_center_fixedVars)
        .def("getStretchingStiffnessFixedVars", &InflatablePeriodicUnit::getStretchingStiffnessFixedVars)
        .def("getBendingStiffnessFixedVars", &InflatablePeriodicUnit::getBendingStiffnessFixedVars)
        // Periodic volume
        .def("periodicVolume", &InflatablePeriodicUnit::periodicVolume)
        .def("update_boundary_triangles", &InflatablePeriodicUnit::update_boundary_triangles)
        .def("energyPeriodicPressurePotential",   &InflatablePeriodicUnit::energyPeriodicPressurePotential)
        .def("gradientPeriodicPressurePotential", &InflatablePeriodicUnit::gradientPeriodicPressurePotential)
        // .def("hessianPeriodicPressurePotential",   py::overload_cast<>(&InflatablePeriodicUnit::hessianPeriodicPressurePotential, py::const_))
        .def("hessianPeriodicPressurePotential", [](const InflatablePeriodicUnit &ipu){ return ipu.hessianPeriodicPressurePotential(); })
        .def("areaFactor", &InflatablePeriodicUnit::areaFactor)
        // Visualization
        .def_property("visualizationTilePower", [](const InflatablePeriodicUnit &ipu) { return ipu.getVisualizationTilePower(); }, [](InflatablePeriodicUnit &ipu, int p) { ipu.setVisualizationTilePower(p); })
        .def("visualizationMesh", &InflatablePeriodicUnit::visualizationMesh, py::arg("duplicateFusedTris") = false)
        .def("visualizationField", &InflatablePeriodicUnit::visualizationField, py::arg("field"), py::arg("duplicateFusedTris") = false)
        // Interface for MeshFEM's viewer
        .def("visualizationGeometry", [](const InflatablePeriodicUnit &ipu, double normalCreaseAngle) { return getVisualizationGeometry(*ipu.visualizationMesh(), normalCreaseAngle); }, py::arg("normalCreaseAngle") =  M_PI)
        // Serialization
        // TODO (Samara)
        // Debug
        .def("get_x_flat", &InflatablePeriodicUnit::get_x_flat)
        .def("get_kappa", &InflatablePeriodicUnit::get_kappa)
        .def("get_alpha", &InflatablePeriodicUnit::get_alpha)
        .def("get_axis_mat", &InflatablePeriodicUnit::get_axis_mat)
        .def("get_grad_omega", &InflatablePeriodicUnit::get_grad_omega)
        .def("set_kappa", &InflatablePeriodicUnit::set_kappa)
        .def("set_alpha", &InflatablePeriodicUnit::set_alpha)
        .def("deactivate_planar_homogenization", &InflatablePeriodicUnit::deactivate_planar_homogenization)
        .def("setHessianColIdentity", &InflatablePeriodicUnit::setHessianColIdentity)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Inflatable Mid Surface Periodic Unit
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<InflatableMidSurfacePeriodicUnit, std::shared_ptr<InflatableMidSurfacePeriodicUnit>> pyInflatableMidSurfacePeriodicUnit(m, "InflatableMidSurfacePeriodicUnit");

    pyInflatableMidSurfacePeriodicUnit
        .def(py::init<const std::shared_ptr<Mesh> &, const std::vector<bool> &, Real>(), py::arg("mesh"), py::arg("fusedVtx") = std::vector<bool>(), py::arg("epsilon") = 1e-7)

        .def("numVars",      &InflatableMidSurfacePeriodicUnit::numVars)
        .def("getVars",      &InflatableMidSurfacePeriodicUnit::getVars)
        .def("setVars",      &InflatableMidSurfacePeriodicUnit::setVars)
        .def("get_average_z_idx", &InflatableMidSurfacePeriodicUnit::get_average_z_idx)
        
        .def("energy",   &InflatableMidSurfacePeriodicUnit::energy  , py::arg("energyType") = IPUEType::Full)
        .def("gradient", &InflatableMidSurfacePeriodicUnit::gradient, py::arg("energyType") = IPUEType::Full)

        .def("hessianSparsityPattern", &InflatableMidSurfacePeriodicUnit::hessianSparsityPattern, py::arg("val") = 1.0)
        .def("hessian",  py::overload_cast<IPUEType, bool>(&InflatableMidSurfacePeriodicUnit::hessian, py::const_), py::arg("energyType") = IPUEType::Full, py::arg("kappaOnly") = false)
        .def("areaFactor", &InflatableMidSurfacePeriodicUnit::areaFactor)
        .def("getRigidMotionFixedVars", &InflatableMidSurfacePeriodicUnit::getRigidMotionFixedVars)
        .def("getStretchingStiffnessFixedVars", &InflatableMidSurfacePeriodicUnit::getStretchingStiffnessFixedVars)
        .def("getBendingStiffnessFixedVars", &InflatableMidSurfacePeriodicUnit::getBendingStiffnessFixedVars)
        .def_readonly("ipu", &InflatableMidSurfacePeriodicUnit::ipu, py::return_value_policy::reference)
        .def("getMidSurfaceToPeriodicPatchMapTranspose_all_vars", &InflatableMidSurfacePeriodicUnit::getMidSurfaceToPeriodicPatchMapTranspose_all_vars, py::return_value_policy::reference)
        // Visualization
        .def("visualizationMesh", &InflatableMidSurfacePeriodicUnit::visualizationMesh, py::arg("duplicateFusedTris") = false)
        .def("visualizationField", &InflatableMidSurfacePeriodicUnit::visualizationField, py::arg("field"), py::arg("duplicateFusedTris") = false)
        // Interface for MeshFEM's viewer
        .def("visualizationGeometry", [](const InflatableMidSurfacePeriodicUnit &ipu, double normalCreaseAngle) { return getVisualizationGeometry(*ipu.visualizationMesh(), normalCreaseAngle); }, py::arg("normalCreaseAngle") =  M_PI)
        ;
   

    ////////////////////////////////////////////////////////////////////////////////
    // Enable output redirection from Python side
    ////////////////////////////////////////////////////////////////////////////////
    py::add_ostream_redirect(m, "ostream_redirect");
}
