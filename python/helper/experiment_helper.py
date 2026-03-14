import os
import sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import homogenized_inflation, numpy as np
import  utils
import periodic_unit_helper
import numpy.linalg as la
from periodic_simulation_setup import *
from tri_mesh_viewer import TriMeshViewer

low_pressure_tag = "low_pressure"
high_pressure_tag = "high_pressure"

render_images = False

hessianShiftForRigidMotion = 1e-7
hessianShiftForAlphainPlanar = 1e-12
hessianShiftForStiffness = 1e-7

def helper_run_equilibrium(ipu, allow_bending, stiffness_pressure, disableFusedRegionTFT = False, useTFT = True, callback = None):
    # Choose strategy for constraining rigid motion
    # We first use no fix vars to check whether the equilibrium converge to a planar state, use a hessian shift rather than applying fixed vars to constrain the rigid motion so that the equilibrium solve converges faster. 
    bending_fixed_vars = [] if allow_bending else [ipu.numVars() - 2]
    fixedVars, hessianShift = [ipu.numVars() - 2], hessianShiftForRigidMotion

    ipu.sheet.setUseTensionFieldEnergy(useTFT)
    ipu.sheet.setUseHessianProjectedEnergy(False)
    if (disableFusedRegionTFT):
        ipu.sheet.disableFusedRegionTensionFieldTheory(False)
    ipu.sheet.pressure = stiffness_pressure

    initial_dofs = ipu.getVars()

    check_gradient = True
    def default_cb(it):
        # Define a callback so it can be interrupted by the user.
        if check_gradient and la.norm(ipu.gradient()) > 1e5:
            print("Gradient exploded.")
            ipu.setVars(initial_dofs)
            hessianShiftForRigidMotion *= 10
        return
    
    experiment_callback = default_cb if (callback is None) else callback

    opts.niter = 100
    opts.gradTol = 1e-10
    cr = homogenized_inflation.inflation_newton(ipu, fixedVars, opts, callback=experiment_callback, hessianShift = hessianShift)

    check_gradient = False

    opts.niter = 100
    # Solve for true equilibrium with a much smaller hessian shift that's only used for the alpha variable when kappa becomes zero. Pin down the the x, y, z value of a center vertex. 
    fixedVars, hessianShift = list(periodic_unit_helper.get_center_fixedVars(ipu)) + [ipu.numVars() - 2], hessianShiftForAlphainPlanar
    cr = homogenized_inflation.inflation_newton(ipu, fixedVars, opts, callback=experiment_callback, hessianShift = hessianShift)

    opts.niter = 500
    # Solve for true equilibrium with a much smaller hessian shift that's only used for the alpha variable when kappa becomes zero. Pin down the the x, y, z value of a center vertex. 
    fixedVars, hessianShift = list(periodic_unit_helper.get_center_fixedVars(ipu)) + bending_fixed_vars, hessianShiftForAlphainPlanar
    cr = homogenized_inflation.inflation_newton(ipu, fixedVars, opts, callback=experiment_callback, hessianShift = hessianShift)
    return cr

def helper_compute_bending_stiffness(ipu, cb, result_folder = None, name = None, variable = None, render_images = False):
    # Compute stiffness after removing the vertical offset
    reparametrize_gamma_bar(ipu)
    stiffness_fixedVars = list(periodic_unit_helper.get_center_fixedVars(ipu)) + [ipu.numVars() - 1, ipu.numVars() - 2]
    optimizer = homogenized_inflation.get_inflation_optimizer(ipu, stiffness_fixedVars, opts, callback=cb, hessianShift = 0)
    stiffness_values, sampled_alphas, stiffness_coefficient = visualize_sampled_bending_stiffness(ipu, 1000, optimizer, hessianShift = hessianShiftForStiffness, fixedVars = stiffness_fixedVars, filename = None if (result_folder is None) else ("{}/stiffness_{}_{}.png".format(result_folder, name, variable)), generate_images = render_images)
    return stiffness_values, sampled_alphas, stiffness_coefficient

def run_experiment(ipu, m, fusedVtx, stiffness_pressure, scale_factor_pressure, name, variable, result_folder, disableFusedRegionTFT = False, useTFT = True, use_low_pressure = False, omit_negative_curvature_escape = False, allow_bending = True):
    experiment_log = {}
    configure_solver_parallelism()

    m.save("{}/mesh_{}_{}.obj".format(result_folder, name, variable))
    np.save("{}/fusedVtx_{}_{}.npy".format(result_folder, name, variable), fusedVtx)
    
    if (render_images):
        viewer = TriMeshViewer(ipu, width=768, height=640)
        viewer.showWireframe(True)

    def cb(it):
        # Define a callback so it can be interrupted by the user.
        return

    cr = helper_run_equilibrium(ipu, allow_bending, stiffness_pressure, disableFusedRegionTFT = disableFusedRegionTFT, useTFT = useTFT)

    experiment_log["Equilibrium energy"] = ipu.energy()
    np.save("{}/{}_dofs_before_stiffness_{}.npy".format(result_folder, name, variable), ipu.getVars())

    # Default to 0.
    experiment_log["Negative stiffness"] = 0

    if np.abs(ipu.getVars()[-2]) > 1e-7:
        # Terminate early because we cannot do stiffness analysis on a non-planar equilibrium.
        experiment_log["Ipu simulation succeed"] = int(cr.success)
        experiment_log["Planar equilibrium"] = 0
        experiment_log["Simulation Kappa value"] = (ipu.getVars()[-2])
        return experiment_log

    stiffness_values, sampled_alphas, stiffness_coefficient = helper_compute_bending_stiffness(ipu, cb, result_folder, name, variable)

    experiment_log["Min bending stiffness"] = np.min(stiffness_values)
    if np.min(stiffness_values) < 0:
        experiment_log["Negative stiffness"] = 1
        curr_vars = ipu.getVars()
        curr_vars[-1] = sampled_alphas[np.argmin(stiffness_values)]
        # Push the surface slightly out of plane to escape from the negative stiffness direction.
        curr_vars[-2] = 0.1
        ipu.setVars(curr_vars)
        # For gathering the homogenization dataset we don't really need to run this step. So we just run 100 iteration to get away from the planar configuration then stop.
        if (omit_negative_curvature_escape):
            opts.niter = 100
    
        bending_fixed_vars = [] if allow_bending else [ipu.numVars() - 2]
        fixedVars, hessianShift = list(periodic_unit_helper.get_center_fixedVars(ipu)) + bending_fixed_vars, hessianShiftForAlphainPlanar
        cr = homogenized_inflation.inflation_newton(ipu, fixedVars, opts, callback=cb, hessianShift = hessianShift)
        experiment_log["Equilibrium energy"] = ipu.energy()
        np.save("{}/{}_dofs_after_negative_stiffness_escape_{}.npy".format(result_folder, name, variable), ipu.getVars())
        
    # Log and save results.
    experiment_log["Ipu simulation succeed"] = int(cr.success)
    experiment_log["Simulation Kappa value"] = (ipu.getVars()[-2])

    if np.abs(ipu.getVars()[-2]) > 1e-7:
        experiment_log["Planar equilibrium"] = 0
        return experiment_log
    else:
        experiment_log["Planar equilibrium"] = 1

    if (render_images):
        viewer.update(scalarField=utils.getStrains(ipu.sheet)[:, 0])
        render = viewer.offscreenRenderer(1000, 1000)
        render.render()
        render.save("{}/{}_render_{}_{}.png".format(result_folder, high_pressure_tag, name, variable))
        visualize_average_deformation_gradient(ipu, 100, filename = "{}/average_deformation_gradient_{}_{}.png".format(result_folder, name, variable))

    print("computing stretching stiffness")
    # Compute stretching stiffness.
    betas = np.linspace(0, 2 * np.pi, 1000)
    optimizer = homogenized_inflation.get_inflation_optimizer(ipu, ipu.getStretchingStiffnessFixedVars(), opts, callback=cb, hessianShift = hessianShiftForStiffness)
    stretchingStiffness = homogenized_inflation.getStretchingStiffness(ipu, betas, optimizer, hessianShiftForStiffness, ipu.getStretchingStiffnessFixedVars())
    np.save("{}/stretching_stiffness_{}_{}.npy".format(result_folder, name, variable), stretchingStiffness)

    np.save("{}/bending_stiffness_values_{}_{}.npy".format(result_folder, name, variable), stiffness_values)
    np.save("{}/sampled_alphas_{}_{}.npy".format(result_folder, name, variable), sampled_alphas)
    np.save("{}/stiffness_coefficient_{}_{}.npy".format(result_folder, name, variable), stiffness_coefficient)
    np.save("{}/experiment_parameters.npy".format(result_folder), np.array([stiffness_pressure, scale_factor_pressure, disableFusedRegionTFT, useTFT]))
    np.save("{}/{}_strain_values_{}_{}.npy".format(result_folder, high_pressure_tag, name, variable), utils.getStrains(ipu.sheet)[:, 0])
    np.save("{}/scale_factors_{}_{}.npy".format(result_folder, name, variable), get_deformation_scale_factors(ipu))
    np.save("{}/average_deformation_gradient_matrix_{}_{}.npy".format(result_folder, name, variable), get_deformation_matrix(ipu))
    np.save("{}/strain_values_{}_{}.npy".format(result_folder, name, variable), utils.getStrains(ipu.sheet)[:, 0])

    ipu.visualizationMesh().save("{}/equilibrium_mesh_{}_{}.obj".format(result_folder, name, variable))
    return experiment_log


# Helper class for saving experiment data in json
# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
import json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
