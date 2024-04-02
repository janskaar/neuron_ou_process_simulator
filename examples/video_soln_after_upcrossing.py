import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from scipy.integrate import cumtrapz
from scipy.stats import multivariate_normal, norm
from scipy.linalg import expm
from scipy.special import erfc, erf
import sys, os
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, MomentsSimulator, MembranePotentialSimulator, ParticleSimulator
from neurosim.n_functions import compute_n1, pdf_b, xv_to_xy, xy_to_xv, integral_f1_xdot_gaussian, integral_f1_xdot

from neurosim.n_functions import compute_p_v_upcrossing, conditional_bivariate_gaussian, gaussian_pdf
from neurosim.n_functions import ou_soln_xv_after_upcrossing, ou_soln_marginal_x_after_upcrossing, ou_soln_marginal_v_after_upcrossing
from neurosim.n_functions import ou_soln_xv_integrand, ou_soln_xv_after_upcrossing_arguments

phi = np.arctan(1080 / 1920)
sz = (14 * np.cos(phi), 14 * np.sin(phi))

## ====================

# Verify second upcrossing probability
# NOTE: The plots look wrong at the very beginning of the simulation,
# but this is a discretization artefact. Increasing the resolution 10-fold,
# the same artefact is still present for the first ~20 time steps, but correct
# after.

# simulate expectation / covariance for first upcrossing
t1 = 10.
p = SimulationParameters(threshold=0.01, dt=0.01, I_e = 0.1, num_procs=100000)
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)
moments_sim = MomentsSimulator(mu_0, s_0, p)
moments_sim.simulate(t1)
mu_xy = moments_sim.mu
s_xy = moments_sim.s
mu_xv, s_xv = xy_to_xv(mu_xy, s_xy, p)

membrane_potential_sim = MembranePotentialSimulator(0., p)
membrane_potential_sim.simulate(t1)
b = membrane_potential_sim.b
b_dot = membrane_potential_sim.b_dot


upcrossing_ind = 9000

# compute p(v) at upcrossing
v_vec = np.linspace(b_dot[upcrossing_ind], 0.1, 10001)
p_v_0 = compute_p_v_upcrossing(v_vec, b[upcrossing_ind], b_dot[upcrossing_ind], mu_xv[upcrossing_ind], s_xv[upcrossing_ind]).squeeze()
p_v_integral = np.trapz(p_v_0, x=v_vec)
print(f"p(v) INTEGRAL: {p_v_integral}")
p_v_0 /= p_v_integral

# simulate with initial conditions of upcrossing
t = 0.1
t_vec = np.arange(0, t+p.dt, p.dt)
z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
v_0 = np.random.choice(v_vec, p=p_v_0 * (v_vec[1] - v_vec[0]), size=p.num_procs, replace=True)
y_0 = (v_0 + b[upcrossing_ind] / p.tau_x) * p.C
z_0[:,0] = y_0
z_0[:,1] = b[upcrossing_ind]
sim = ParticleSimulator(z_0, 0., p)
sim.simulate(t)
xv = np.zeros_like(sim.z)
xv[...,1] = sim.z[...,1]
xv[...,0] = -xv[...,1] / p.tau_x + sim.z[...,0] / p.C


##

# Create different linspaces per time step, based on values
# from simulation
xmin = xv[...,1].min(1)
xmax = xv[...,1].max(1)
xdiff = xmax - xmin
xmin = xmin - 0.25 * xdiff
xmax = xmax + 0.25 * xdiff

vmin = xv[...,0].min(1)
vmax = xv[...,0].max(1)
vdiff = vmax - vmin
vmin = vmin - 0.25 * vdiff
vmax = vmax + 0.25 * vdiff

def veclinspace(a, b, n):
    return jnp.linspace(a, b, n)

veclinspace = jax.vmap(veclinspace, in_axes=(0, 0, None))
num_vals = 501
v_vecs = veclinspace(vmin, vmax, num_vals)
x_vecs = veclinspace(xmin, xmax, num_vals)

#
# Vectorize solution functions
#

# create grid for each time step
def create_grid(v_vec, x_vec):
    vv, xx = jnp.meshgrid(v_vec, x_vec, indexing="ij")
    grid = jnp.stack((vv, xx), axis=-1).reshape((-1, 2))
    return grid

create_grid = jax.vmap(create_grid, in_axes=(0, 0))
xv_grids = create_grid(v_vecs, x_vecs)

# take matrix of (N, 2) xv-values
soln_fn_xv = jax.vmap(ou_soln_xv_after_upcrossing, in_axes=(0, None, None, None, None, None, None))

# multiple time steps for (T, N, 2) xv-values, (T,) b-values, (T,) b_dot-values, (T,) t-values
soln_fn_xv = jax.vmap(soln_fn_xv, in_axes=(0, None, None, None, None, 0, None))

# multiple time steps for (T,) x-values, (T,) b-values, (T,) b_dot-values, (T,) t-values
soln_fn_x = jax.vmap(ou_soln_marginal_x_after_upcrossing, in_axes=(0, None, None, None, None, 0, None))

# multiple time steps for (T,) x-values, (T,) b-values, (T,) b_dot-values, (T,) t-values
soln_fn_v = jax.vmap(ou_soln_marginal_v_after_upcrossing, in_axes=(0, None, None, None, None, 0, None))

p_xv = soln_fn_xv(xv_grids,
              mu_xy[upcrossing_ind],
              s_xv[upcrossing_ind],
              b[upcrossing_ind],
              b_dot[upcrossing_ind],
              t_vec,
              p).squeeze()

p_x = soln_fn_x(x_vecs,
                mu_xy[upcrossing_ind],
                s_xv[upcrossing_ind],
                b[upcrossing_ind],
                b_dot[upcrossing_ind],
                t_vec,
                p).squeeze()
 
p_v = soln_fn_v(v_vecs,
                mu_xy[upcrossing_ind],
                s_xv[upcrossing_ind],
                b[upcrossing_ind],
                b_dot[upcrossing_ind],
                t_vec,
                p).squeeze()


## 

for i in range(100):
    gs = GridSpec(5, 5)
    fig = plt.figure()
    fig.set_size_inches(sz)
    ax1 = fig.add_subplot(gs[:4,:4])
    ax1.scatter(xv[i,:,1], xv[i,:,0], s=1.)
    maxval = p_xv[i].max()
    levels = [i * maxval for i in [0.0001, 0.001, 0.01, 0.1, 0.9]]
    ax1.contour(x_vecs[i], v_vecs[i], p_xv[i].reshape((num_vals, num_vals)), colors="C1", levels=levels)
    
    ax2 = fig.add_subplot(gs[:4, 4])
    ax2.plot(p_v[i], v_vecs[i])
    ax2.set_xticks([])
    
    ax3 = fig.add_subplot(gs[4, :4])
    ax3.plot(x_vecs[i], p_x[i])
    ax3.set_yticks([])
    
    ax1.sharey(ax2)
    ax1.sharex(ax3)
    ax1.set_xlim(x_vecs[i,0], x_vecs[i,-1])
    ax1.set_ylim(v_vecs[i,0], v_vecs[i,-1])
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    ax1.set_ylabel("$\dot{x}$")
    ax3.set_xlabel("$x$")
    ax3.set_ylabel("$p(x)$")
    ax2.set_xlabel("$p(\dot{x})$")
    ax1.set_title("$p(\dot{x}, x)$")
    
    fig.savefig(os.path.join("soln_after_upcrossing_video", f"p_xv_after_upcrossing_{i}.png"))
    plt.close()

