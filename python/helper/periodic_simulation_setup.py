
import homogenized_inflation, numpy as np, importlib, fd_validation, visualization, parametric_pillows, wall_generation
import periodic_unit_helper
from numpy.linalg import norm
import numpy.linalg as la
import MeshFEM, parallelism, benchmark, utils
import igl

from matplotlib import pyplot as plt

import sys; sys.path.append('../../gmsh/')
import pattern_generator_using_gmsh


import py_newton_optimizer
opts = py_newton_optimizer.NewtonOptimizerOptions()
opts.useIdentityMetric = True
opts.beta = 1e-4
opts.gradTol = 1e-11

def configure_solver_parallelism():
    solver = 1
    if solver == 0:
        opts.factorizer = opts.factorizer.PARDISO
        import os; os.environ['OMP_NUM_THREADS'] = '8'
    elif solver == 1:
        opts.factorizer = opts.factorizer.CatamariNesdis
        import os; os.environ['OMP_NUM_THREADS'] = '1'
        parallelism.set_max_num_tbb_threads(8)
    elif solver == 2:
        opts.factorizer = opts.factorizer.CHOLMOD
        import os; os.environ['OMP_NUM_THREADS'] = '1'
        parallelism.set_max_num_tbb_threads(8)


from tri_mesh_viewer import TriMeshViewer

import time, vis

from io_redirection import suppress_stdout

import mode_viewer

import compute_vibrational_modes
class ModalAnalysisWrapper:
    def __init__(self, sheet):
        self.sheet = sheet
    def hessian(self):
        return self.sheet.hessian(self.sheet.EnergyType.Elastic)


def get_fusedVtx_using_markers(numVxs, markers):
    fusedVtx = np.array([False] * numVxs)
    for i in markers:
        fusedVtx[i] = True
    return fusedVtx

class bent_sheet_wrapper:
    def __init__(self, ipu):
        self.ipu = ipu

    def setVars(self, v):
        self.ipu.bent_sheet_setVars(v)
    def numVars(self):
        return self.ipu.bent_sheet_numVars()

    def getVars(self):
        return self.ipu.bent_sheet_getVars()

    def energy(self):   return self.ipu.energy()
    def gradient(self): return self.ipu.bent_sheet_gradient()
    def hessian(self): return self.ipu.bent_sheet_hessian()

class periodic_volume_wrapper:
    def __init__(self, ipu):
        self.ipu = ipu

    def setVars(self, v):
        self.ipu.sheet.setVars(v)
    def numVars(self):
        return self.ipu.sheet.numVars()

    def getVars(self):
        return self.ipu.sheet.getVars()

    def energy(self):   return self.ipu.energyPeriodicPressurePotential()
    def gradient(self): return self.ipu.gradientPeriodicPressurePotential()    
    def hessian(self): return self.ipu.hessianPeriodicPressurePotential()

import scipy

def bending_stiffness_from_hessian(hessian):
    H = periodic_unit_helper.getNumpyArrayFromCSC(hessian, reflect = True)
    H_F_u = scipy.sparse.csr_matrix(H[:-2, :-2])
    d_Fu_d_kappa = H[:-2, -2]
    d_Fu_d_epsilon = scipy.sparse.linalg.spsolve(H_F_u, -d_Fu_d_kappa)
    return H[-2, -2] + d_Fu_d_kappa.dot(d_Fu_d_epsilon)

def bending_stiffness_python(ipu):
    return bending_stiffness_from_hessian(ipu.hessian())

def get_sampled_bending_stiffness_python(ipu, resolution, minAlpha, maxAlpha):
    bending_stiffness_sample_alpha = np.zeros(resolution)
    sampled_alpha = np.linspace(minAlpha, maxAlpha, resolution)
    hessians = []
    curr_vars = ipu.getVars()
    for alpha in sampled_alpha:
        curr_vars[-1] = alpha
        ipu.setVars(curr_vars)
        hessians.append(ipu.hessian())
    for i in range(resolution):
        bending_stiffness_sample_alpha[i] = bending_stiffness_from_hessian(hessians[i])
    return bending_stiffness_sample_alpha, sampled_alpha

def get_sampled_bending_stiffness(ipu, resolution, optimizer, minAlpha, maxAlpha, hessianShift = 1e-10, fixedVars = np.array([])):
    sampled_alpha = np.linspace(minAlpha, maxAlpha, resolution)
    bending_stiffness_sample_alpha = homogenized_inflation.getBendingStiffness(ipu, sampled_alpha, optimizer, hessianShift, fixedVars)
    return bending_stiffness_sample_alpha, sampled_alpha

def get_sampled_bending_stiffness_using_bases(ipu, resolution, optimizer, minAlpha, maxAlpha, hessianShift = 1e-10, fixedVars = np.array([])):
    sampled_alpha = np.linspace(minAlpha, maxAlpha, resolution)
    bending_stiffness_sample_alpha, stiffness_coefficient = homogenized_inflation.getBendingStiffnessUsingBases(ipu, sampled_alpha, optimizer, hessianShift, fixedVars)
    return bending_stiffness_sample_alpha, sampled_alpha, stiffness_coefficient

import numpy as np
import matplotlib.pyplot as plt

def visualize_sampled_bending_stiffness(ipu, resolution, optimizer, minAlpha = 0.0, maxAlpha = np.pi, filename = "stiffness.png", hessianShift = 1e-10, fixedVars = np.array([]), use_bases = True, plot_min_r = None, plot_max_r = None, show_figure = False, generate_images = False):
    stiffness_coefficient = []
    if use_bases:
        sampled_stiffness, sampled_alpha, stiffness_coefficient = get_sampled_bending_stiffness_using_bases(ipu, resolution, optimizer, minAlpha, maxAlpha, hessianShift, fixedVars)
    else:
        sampled_stiffness, sampled_alpha = get_sampled_bending_stiffness(ipu, resolution, optimizer, minAlpha, maxAlpha, hessianShift, fixedVars)
    
    if (not generate_images):
        return sampled_stiffness, sampled_alpha, stiffness_coefficient

    r = list(sampled_stiffness) + list(sampled_stiffness)
    theta = list(sampled_alpha) + list(np.pi + np.array(sampled_alpha))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r)
    ax.set_rmax(max(sampled_stiffness) if plot_max_r is None else plot_max_r)
    ax.set_rmin(min(sampled_stiffness) - 0.2 * (max(sampled_stiffness) - min(sampled_stiffness)) if plot_min_r is None else plot_min_r)
    # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    ax.set_title("Bending stiffness", va='bottom')
    plt.tight_layout()
    plt.savefig(filename, dpi = 300)
    if (not show_figure):
        plt.close()
    return sampled_stiffness, sampled_alpha, stiffness_coefficient

def allEnergies(ipu):
    return {name: ipu.energy(energyType = iet) for name, iet in ipu.EnergyType.__members__.items()}

def allGradientNorms(ipu):
    return {name: la.norm(ipu.gradient(energyType = iet)) for name, iet in ipu.EnergyType.__members__.items()}

# This function does the same thing as inflation.InflatableMidSurfacePeriodicUnit.reparametrize_vertical_offset. 
# Deprecated.
def reparametrize_gamma_bar(ipu):
    gamma = np.mean(ipu.get_x_flat().reshape(int(ipu.sheet.numVars() / 3), 3)[:, 2])
    # New kappa.
    kappa_gamma = ipu.get_kappa()
    kappa_zero = kappa_gamma / (1. - kappa_gamma * gamma)


    alpha_gamma = ipu.get_alpha()
    axis_perp = np.array([-np.sin(alpha_gamma), np.cos(alpha_gamma)])
    F = ipu.getVars()[:3]

    F_mat = np.zeros((2, 2))

    F_mat[0, 0] = F[0]
    F_mat[1, 1] = F[1]
    F_mat[0, 1] = F[2]
    F_mat[1, 0] = F[2]

    scale_factor_mat = np.eye(2) - gamma * kappa_gamma * axis_perp.reshape((2, 1)) * axis_perp.reshape((1, 2))
    # F_zero_general is not symmetric in general. Apply polar decomposition to get a symmetric F_zero.
    rotation, F_zero = scipy.linalg.polar(scale_factor_mat @ F_mat)

    angle = np.arctan2(rotation[1, 0], rotation[0, 0])
    alpha_zero = alpha_gamma - angle

    inverse_rotation = np.array([[np.cos(-angle), -np.sin(-angle), 0], [np.sin(-angle), np.cos(-angle), 0], [0, 0, 1]])
    fluctuation = ipu.getVars()[3:-2].reshape((int((ipu.numVars() - 5) / 3), 3))
    fluctuation[:, 2] -= gamma
    scale_factor_sym_3D = np.eye(3)
    scale_factor_sym_3D[:2, :2] = scale_factor_mat
    fluctuation = (scale_factor_sym_3D @ fluctuation.transpose()).transpose()
    fluctuation = (inverse_rotation @ fluctuation.transpose()).transpose()

    assert np.abs(F_zero[0, 1] - F_zero[1, 0]) < 1e-6

    curr_vars = ipu.getVars()
    curr_vars[0] = F_zero[0, 0]
    curr_vars[1] = F_zero[1, 1]
    curr_vars[2] = F_zero[0, 1]
    curr_vars[-2] = kappa_zero
    curr_vars[-1] = alpha_zero
    curr_vars[3:-2] = fluctuation.flatten()

    ipu.setVars(curr_vars)

def get_deformation_scale_factors(ipu):
    f1, f2, f3 = ipu.getVars()[:3]
    mat = np.array([[f1, f3],
                  [f3, f2]])
    eigs = la.eig(mat)[0]
    return min(eigs), max(eigs)

class stretching_stiffness_class:
    def __init__(self, ipu, sheet, optimizer, viewer, index):
        self.ipu = ipu
        self.sheet = sheet
        self.viewer = viewer
        self.optimizer = optimizer
        self.index = index
        self.factor = ipu.areaFactor()

    def setVars(self, x):
        curr_vars = self.ipu.getVars()
        curr_vars[self.index] = x[0]
        self.ipu.setVars(curr_vars)

        opts.niter = 1000
        opts.gradTol = 1e-11

        framerate = 5 # Update every 5 iterations
        def cb(it):
            if it % framerate == 0:
                self.viewer.update(scalarField=utils.getStrains(self.sheet)[:, 0])
        optimizer = homogenized_inflation.get_inflation_optimizer(self.ipu, self.ipu.getStretchingStiffnessFixedVars(), opts, callback=cb, hessianShift = 0)
        cr = optimizer.optimize()
        print(cr.success)

    def numVars(self):
        return 1

    def getVars(self):
        return self.ipu.getVars()[[self.index]]

    # The stiffness is computed as second derivative density.
    def energy(self): 
        return self.ipu.energy() / self.factor

    def gradient(self):
        return self.ipu.gradient()[[self.index]] / self.ipu.areaFactor()
    
    def secondDerivative(self): return homogenized_inflation.debugTangentElasticityTensor(self.ipu, [], self.optimizer, 0, self.ipu.getStretchingStiffnessFixedVars())[self.index][self.index]
    
class bending_stiffness_class:
    # This class can only be defined on mid surface periodic unit. Even though we can compute bending stiffness at zero vertical offset and planar state, we cannot run fd validation on it because we need to evaluate at non zero kappa.
    def __init__(self, ipu, sheet, optimizer, viewer, fixedVars = np.array([])):
        self.ipu = ipu
        self.sheet = sheet
        self.viewer = viewer
        self.optimizer = optimizer
        self.fixedVars = fixedVars
        self.factor = ipu.areaFactor()

    def setVars(self, kappa):
        curr_vars = self.ipu.getVars()
        curr_vars[-2] = kappa[0]
        self.ipu.setVars(curr_vars)
        fixedVars = [self.ipu.get_average_z_idx(), periodic_unit_helper.get_center_fixedVars(self.ipu.ipu)[0], periodic_unit_helper.get_center_fixedVars(self.ipu.ipu)[1], self.ipu.numVars() - 2, self.ipu.numVars() - 1]

        opts.niter = 1000
        opts.gradTol = 1e-11

        framerate = 5 # Update every 5 iterations
        def cb(it):
            if it % framerate == 0:
                self.viewer.update(scalarField=utils.getStrains(self.sheet)[:, 0])
        optimizer = homogenized_inflation.get_inflation_optimizer(self.ipu, fixedVars, opts, callback=cb, hessianShift = 0)
        cr = optimizer.optimize()
        print(cr.success)

    def numVars(self):
        return 1

    def getVars(self):
        return np.array([self.ipu.getVars()[-2]])

    # The stiffness is computed as second derivative density.
    def energy(self): return self.ipu.energy() / self.factor


    def secondDerivative(self): return homogenized_inflation.getBendingStiffness(self.ipu, np.array([self.ipu.getVars()[-1]]), self.optimizer, 0, [self.ipu.get_average_z_idx(), periodic_unit_helper.get_center_fixedVars(self.ipu.ipu)[0], periodic_unit_helper.get_center_fixedVars(self.ipu.ipu)[1], self.ipu.numVars() - 2, self.ipu.numVars() - 1])


class bending_total_gradient:
    def __init__(self, ipu, sheet, optimizer, viewer):
        self.ipu = ipu
        self.sheet = sheet
        self.viewer = viewer
        self.optimizer = optimizer

    def setVars(self, kappa):
        curr_vars = self.ipu.getVars()
        curr_vars[-2] = kappa[0]
        self.ipu.setVars(curr_vars)
        fixedVars, hessianShift = [self.ipu.numVars() - 2, self.ipu.numVars() - 1], 1e-6
        opts.niter = 1000
        opts.gradTol = 1e-11

        framerate = 5 # Update every 5 iterations
        def cb(it):
            if it % framerate == 0:
                self.viewer.update(scalarField=utils.getStrains(self.sheet)[:, 0])
        optimizer = homogenized_inflation.get_inflation_optimizer(self.ipu, fixedVars, opts, callback=cb, hessianShift = hessianShift)
        cr = optimizer.optimize()
        print(cr.success)

    def numVars(self):
        return 1

    def getVars(self):
        return np.array([self.ipu.getVars()[-2]])

    def energy(self):   return self.ipu.energy()
    def gradient(self): return np.array(self.ipu.gradient()[-2])

class bending_test_class:
    def __init__(self, ipu):
        self.ipu = ipu

    def setVars(self, v0):
        curr_vars = self.ipu.getVars()
        curr_vars[0] = v0[0]
        self.ipu.setVars(curr_vars)
       
    def numVars(self):
        return 1

    def getVars(self):
        return np.array([self.ipu.getVars()[0]])

    def energy(self):   return self.ipu.energy()
    def secondDerivative(self): return self.ipu.hessian().Ax[0]

def get_az_ipu_from_ipu(ipu, m, fusedVtx, useTFT, disableFusedRegionTFT = False):
    ''' ipu will be reparametrized to have zero average z offset. 
    '''
    reparametrize_gamma_bar(ipu)
    az_ipu = homogenized_inflation.InflatableMidSurfacePeriodicUnit(m, fusedVtx, epsilon = 1e-9)
    az_ipu.ipu.setVars(ipu.getVars())
    az_ipu.ipu.sheet.setUseTensionFieldEnergy(useTFT)
    az_ipu.ipu.sheet.setUseHessianProjectedEnergy(False)
    if disableFusedRegionTFT:
        az_ipu.ipu.sheet.disableFusedRegionTensionFieldTheory(False)
    az_ipu.ipu.sheet.pressure = ipu.sheet.pressure
    return az_ipu

def get_deformation_matrix(ipu):
    F = ipu.getVars()[:3]

    F_mat = np.zeros((2, 2))

    F_mat[0, 0] = F[0]
    F_mat[1, 1] = F[1]
    F_mat[0, 1] = F[2]
    F_mat[1, 0] = F[2]
    return F_mat
    
def visualize_average_deformation_gradient(ipu, resolution, minAlpha = 0.0, maxAlpha = 2 * np.pi, filename = "average_deformation_gradient.png", plot_min_r = None, plot_max_r = None, show_figure = False, inverse = False):
    F_mat = get_deformation_matrix(ipu)
    
    if inverse:
        F_mat = la.inv(F_mat)
    sampled_alpha = np.linspace(minAlpha, maxAlpha, resolution)
    
    r = []
    theta = []
    final_points = []
    for alpha in sampled_alpha:
        point = np.array([np.cos(alpha), np.sin(alpha)])
        transformed_point = F_mat @ point
        theta.append(np.arctan2(transformed_point[1], transformed_point[0]))
        r.append(np.linalg.norm(transformed_point))
        final_points.append(transformed_point)
        

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r)
    ax.set_rmax(max(r) if plot_max_r is None else plot_max_r)
    ax.set_rmin(min(r) - 0.2 * (max(r) - min(r)) if plot_min_r is None else plot_min_r)
    # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    ax.set_title("Average deformation gradient", va='bottom')
    plt.tight_layout()
    plt.savefig(filename, dpi = 300)
    if (not show_figure):
        plt.close()
    return np.array(final_points)