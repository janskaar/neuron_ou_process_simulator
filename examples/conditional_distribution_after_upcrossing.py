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

p = SimulationParameters(threshold=0.01, dt=0.01, I_e = 0.1, num_procs=1000000)

## ====================

# NOTE: The plots look wrong at the very beginning of the simulation,
# but this is a discretization artefact. Increasing the resolution 10-fold,
# the same artefact is still present for the first ~20 time steps, but correct
# after.

# simulate expectation / covariance for first upcrossing
t1_sim = 10.
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)
msim = MomentsSimulator(mu_0, s_0, p)
msim.simulate(t1_sim)
mu_xy = msim.mu
s_xy = msim.s
mu_xv, s_xv = xy_to_xv(mu_xy, s_xy, p)

usim = MembranePotentialSimulator(0., p)
usim.simulate(t1_sim)
b = usim.b
b_dot = usim.b_dot

upcrossing_ind = 900
 
##

# # # # # # # # # # # # # # # # # # # # # # #
# Verify conditional distribution at time t1 
# given upcrossing distribution at time t0
# # # # # # # # # # # # # # # # # # # # # # #

# compute p(v) at upcrossing
v_vec_upcrossing = np.linspace(b_dot[upcrossing_ind], 0.1, 10001)
p_v_0 = compute_p_v_upcrossing(v_vec_upcrossing, b[upcrossing_ind], b_dot[upcrossing_ind], mu_xv[upcrossing_ind], s_xv[upcrossing_ind]).squeeze()
p_v_integral = np.trapz(p_v_0, x=v_vec_upcrossing)
print(f"p(v) INTEGRAL: {p_v_integral}")
p_v_0 /= p_v_integral

z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
v_0 = np.random.choice(v_vec_upcrossing, p=p_v_0 * (v_vec_upcrossing[1] - v_vec_upcrossing[0]), size=p.num_procs, replace=True)
y_0 = (v_0 + b[upcrossing_ind] / p.tau_x) * p.C
z_0[:,0] = y_0
z_0[:,1] = b[upcrossing_ind]

# simulate with initial conditions of upcrossing
t = 1.
t_vec = np.arange(0, t+p.dt, p.dt)
sim = ParticleSimulator(z_0, 0., p)
sim.simulate(t)
xv = np.zeros_like(sim.z)
xv[...,1] = sim.z[...,1]
xv[...,0] = -xv[...,1] / p.tau_x + sim.z[...,0] / p.C

t_ind = 25

vmin = xv[t_ind,:,0].min()
vmax = xv[t_ind,:,0].max()
vdiff = vmax - vmin
vmin = vmin - 0.25 * vdiff
vmax = vmax + 0.25 * vdiff

xval = 0.007 # x-value to condition v on

v_vec = np.linspace(vmin, vmax, 501)

z_arr = np.stack((v_vec, np.zeros_like(v_vec) + xval), axis=-1)

#
# Vectorize solution functions
#

# take matrix of (N, 2) xv-values
soln_fn_xv = jax.vmap(ou_soln_xv_after_upcrossing, in_axes=(0, None, None, None, None, None, None))

soln_fn_xv_args = jax.vmap(ou_soln_xv_after_upcrossing_arguments, in_axes=(0, None, None, None, None, None, None))

p_xv = soln_fn_xv(z_arr,
                  mu_xv[upcrossing_ind],
                  s_xv[upcrossing_ind],
                  b[upcrossing_ind],
                  b_dot[upcrossing_ind],
                  t_vec[t_ind],
                  p).squeeze()

q, quad = soln_fn_xv_args(z_arr,
                          mu_xv[upcrossing_ind],
                          s_xv[upcrossing_ind],
                          b[upcrossing_ind],
                          b_dot[upcrossing_ind],
                          t_vec[t_ind],
                          p)

p_x = ou_soln_marginal_x_after_upcrossing(xval,
                                          mu_xv[upcrossing_ind],
                                          s_xv[upcrossing_ind],
                                          b[upcrossing_ind],
                                          b_dot[upcrossing_ind],
                                          t_vec[t_ind],
                                          p)

## 

inds = (xv[t_ind,:,1] > xval - 0.00001) & (xv[t_ind,:,1] < xval + 0.00001)

plt.plot(v_vec, p_xv / p_x)
plt.hist(xv[t_ind,inds,0], bins=50, histtype="step", color="C1", density=True)

plt.show()

##

