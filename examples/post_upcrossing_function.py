import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jax
import jax.numpy as jnp
import jax.scipy as jsp
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

upcrossing_ind = 500

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

t_ind = 10

vmin = xv[t_ind,:,0].min()
vmax = xv[t_ind,:,0].max()
vdiff = vmax - vmin
vmin = vmin - 0.25 * vdiff
vmax = vmax + 0.25 * vdiff

xmin = xv[t_ind,:,1].min()
xmax = xv[t_ind,:,1].max()
xdiff = xmax - xmin
xmin = xmin - 0.25 * xdiff
xmax = xmax + 0.25 * xdiff

v_vec = np.linspace(vmin, vmax, 501)
x_vec = np.linspace(xmin, xmax, 501)

vv, xx = np.meshgrid(v_vec, x_vec)
z_arr = np.stack((vv, xx), axis=-1).reshape((-1, 2))

#
# Vectorize solution functions
#

# take matrix of (N, 2) xv-values
soln_fn_xv = jax.vmap(ou_soln_xv_after_upcrossing, in_axes=(0, None, None, None, None, None, None))

soln_fn_xv_args = jax.vmap(ou_soln_xv_after_upcrossing_arguments, in_axes=(0, None, None, None, None, None, None))

p_v = ou_soln_marginal_v_after_upcrossing(v_vec, mu_xv[upcrossing_ind], s_xv[upcrossing_ind], b[upcrossing_ind], b_dot[upcrossing_ind], t_vec[t_ind], p)

p_x = ou_soln_marginal_x_after_upcrossing(x_vec, mu_xv[upcrossing_ind], s_xv[upcrossing_ind], b[upcrossing_ind], b_dot[upcrossing_ind], t_vec[t_ind], p)

p_xv = soln_fn_xv(z_arr,
                  mu_xv[upcrossing_ind],
                  s_xv[upcrossing_ind],
                  b[upcrossing_ind],
                  b_dot[upcrossing_ind],
                  t_vec[t_ind],
                  p)

q, quad, prefactor, alpha, beta, q_prefactor, S_inv = soln_fn_xv_args(z_arr,
                                                               mu_xv[upcrossing_ind],
                                                               s_xv[upcrossing_ind],
                                                               b[upcrossing_ind],
                                                               b_dot[upcrossing_ind],
                                                               t_vec[t_ind],
                                                               p)

alpha = alpha[0]
beta = beta[0]
q = q.reshape((len(v_vec), len(x_vec)))
quad = quad.reshape((len(v_vec), len(x_vec)))
prefactor = prefactor[0]
q_prefactor = q_prefactor[0]
S_inv = S_inv[0]
p_xv = p_xv.reshape((len(v_vec), len(x_vec)))

## 
# # # # # # # # # # # # # # # # # # # # # # # #
# 
# # # # # # # # # # # # # # # # # # # # # # # #

ind = 50
q_ = q[ind]
quad_ = quad[ind]
p_xv_ = p_xv[ind]

f = np.exp(quad_) - np.sqrt(np.pi) * q_ * np.exp(q_ ** 2 + quad_) * erfc(q_)
g = np.sqrt(np.pi) * q_ * np.exp(q_ ** 2) * erfc(q_)
g = np.sqrt(np.pi) * q_ * np.exp(q_ ** 2) * erfc(q_)
dg_dq = 2 * q * g + g / q - 2 * q
# Prefactor is off by a factor 2 as it's computed in a slightly different form from the equation 
dq_dv = -(alpha[0] * S_inv[0,0] + alpha[1] * S_inv[1,0]) * q_prefactor * 2

# MU WRONG, IT'S A CONDITIONAL MEAN!!!!
mu_ = alpha[0] * b_dot[upcrossing_ind] + beta[0]
#line = 



term1 = np.exp(quad_) * prefactor
term2 = -np.sqrt(np.pi) * q_ * np.exp(q_ ** 2 + quad_) * erfc(q_) * prefactor

plt.plot(v_vec, f, color="black", lw=2.)
plt.plot(v_vec, np.exp(quad_) * (1 - g[ind]))
# plt.plot(v_vec, np.exp(quad_))
# plt.plot(v_vec, np.sqrt(np.pi) * q_ * np.exp(q_ ** 2 + quad_) * erfc(q_))
plt.twinx()
plt.plot(v_vec, q_, "--")
plt.plot(v_vec, q_ ** 2, "--")
plt.plot(v_vec, quad_, "--")
plt.ylim(-10, 10)
plt.show()

##

a = np.linspace(-5, 10, 1001)
plt.plot(a, 1 - np.sqrt(np.pi) * a * np.exp(a ** 2) * erfc(a), color="black", lw=2.)

e_ = np.e * np.sqrt(np.pi) * erfc(1)
#plt.plot(a, (1 - e_) + (2 - 3 * e_) * (a - 1) + (4 - 5 * e_) * (a - 1) ** 2)
#plt.plot(a, 1 / (2 * a ** 2))
plt.plot(a, -2 * np.sqrt(np.pi) * a * np.exp(a ** 2))
plt.plot(a, -np.exp(a))
plt.xlim(-5, 5)
plt.ylim(0, 10)

plt.show()

## 
# # # # # # # # # # # # # # # # # # # # # # # #
# Verify that the joint distribution is correct
# # # # # # # # # # # # # # # # # # # # # # # #

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(xv[t_ind,:,0], xv[t_ind,:,1], s=1.)
levels = [100, 1000, 10000, 100000, 180000]
c = ["C1"]
ax.contour(vv, xx, p_xv.reshape((len(v_vec), len(x_vec))), levels=levels, colors=c)
plt.show()

