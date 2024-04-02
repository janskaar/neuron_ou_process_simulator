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

t1_sim = 10.
p = SimulationParameters(threshold=0.01, dt=0.001, I_e = 0.1, num_procs=100000)
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

upcrossing_ind_1 = 9000

# compute p(v) at upcrossing
v_vec = np.linspace(b_dot[upcrossing_ind_1], 0.1, 10001)
p_v_0 = compute_p_v_upcrossing(v_vec, b[upcrossing_ind_1], b_dot[upcrossing_ind_1], mu_xv[upcrossing_ind_1], s_xv[upcrossing_ind_1]).squeeze()
p_v_integral = np.trapz(p_v_0, x=v_vec)
print(f"p(v) INTEGRAL: {p_v_integral}")
p_v_0 /= p_v_integral

# simulate with initial conditions of upcrossing
t = 0.1
t_vec = np.arange(0, t+p.dt, p.dt)
z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
v_0 = np.random.choice(v_vec, p=p_v_0 * (v_vec[1] - v_vec[0]), size=p.num_procs, replace=True)
y_0 = (v_0 + b[upcrossing_ind_1] / p.tau_x) * p.C
z_0[:,0] = y_0
z_0[:,1] = b[upcrossing_ind_1]
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
              mu_xy[upcrossing_ind_1],
              s_xv[upcrossing_ind_1],
              b[upcrossing_ind_1],
              b_dot[upcrossing_ind_1],
              t_vec,
              p).squeeze()

p_x = soln_fn_x(x_vecs,
                mu_xy[upcrossing_ind_1],
                s_xv[upcrossing_ind_1],
                b[upcrossing_ind_1],
                b_dot[upcrossing_ind_1],
                t_vec,
                p).squeeze()
 
p_v = soln_fn_v(v_vecs,
                mu_xy[upcrossing_ind_1],
                s_xv[upcrossing_ind_1],
                b[upcrossing_ind_1],
                b_dot[upcrossing_ind_1],
                t_vec,
                p).squeeze()

## 

# # compute p(v) at upcrossing
# v_vec = np.linspace(b_dot[upcrossing_ind_1], 0.1, 10001)
# p_v_0 = compute_p_v_upcrossing(v_vec, b[upcrossing_ind_1], b_dot[upcrossing_ind_1], mu_xv[upcrossing_ind_1], s_xv[upcrossing_ind_1]).squeeze()
# p_v_integral = np.trapz(p_v_0, x=v_vec)
# print(f"p(v) INTEGRAL: {p_v_integral}")
# p_v_0 /= p_v_integral
# 
# p = SimulationParameters(threshold=0.01, dt=0.01, I_e = 0.1, num_procs=500000)
# z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
# #v_0 = np.random.choice(v_vec, p=p_v_0 * (v_vec[1] - v_vec[0]), size=p.num_procs, replace=True)
# y_0 = (v_vec[400] + b[upcrossing_ind_1] / p.tau_x) * p.C
# z_0[:,0] = y_0
# z_0[:,1] = b[upcrossing_ind_1]
# 
# # simulate with initial conditions of upcrossing
# t = 1.
# t_vec = np.arange(0, t+p.dt, p.dt)
# sim = ParticleSimulator(z_0, 0., p)
# sim.simulate(t)
# xv = np.zeros_like(sim.z)
# xv[...,1] = sim.z[...,1]
# xv[...,0] = -xv[...,1] / p.tau_x + sim.z[...,0] / p.C
# 
# tind = 50
# 
# vmin, vmax = xv[plot_ind,:,0].min(), xv[plot_ind,:,0].max()
# xmin, xmax = xv[plot_ind,:,1].min(), xv[plot_ind,:,1].max()
# 
# v_vec = np.linspace(vmin, vmax, 101)
# x_vec = np.linspace(xmin, xmax, 101)
# vv, xx = np.meshgrid(v_vec, x_vec)
# z_arr = np.stack((vv, xx), axis=-1).reshape((-1, 2))
# 
# xv2 = xv[plot_ind]
# 
# soln_fn = jax.vmap(ou_soln_xv_integrand, in_axes=(0, None, None, None, None, None, None, None))
# f = soln_fn(z_arr,
#             mu_xy[upcrossing_ind],
#             s_xv[upcrossing_ind],
#             v_0,
#             p.threshold,
#             0.,
#             t_vec[plot_ind],
#             p)
# 
# vs = np.linspace(-0.05, 0.05, 1001)
# 
# xval = 0.005
# 
# integrands = integrand_fn(vs,
#                           xval,
#                           v_vec, 
#                           mu_xy[upcrossing_ind_1],
#                           s_xv[upcrossing_ind_1],
#                           b[upcrossing_ind_1],
#                           b_dot[upcrossing_ind_1],
#                           t_vec[tind],
#                           p)
# 
# z = np.stack((vs, np.zeros_like(vs) + xval), axis=-1)
# p_xv = soln_fn_xv(z,
#                   mu_xy[upcrossing_ind_1],
#                   s_xv[upcrossing_ind_1],
#                   b[upcrossing_ind_1],
#                   b_dot[upcrossing_ind_1],
#                   t_vec[tind],
#                   p).squeeze()
# 
# inds = (xv[tind,:,1] > xval - 0.002) & (xv[tind,:,1] < xval + 0.002)
# 
# plt.plot(vs, integrands[400])
# plt.twinx()
# plt.hist(xv[tind,inds,0], bins=50, histtype="step", color="C1");
# plt.show()

## 

index = 50

gs = GridSpec(5, 5)
fig = plt.figure()
fig.set_size_inches(sz)
ax1 = fig.add_subplot(gs[:4,:4])
ax1.scatter(xv[index,:,1], xv[index,:,0], s=1.)
maxval = p_xv[index].max()
levels = [i * maxval for i in [0.0001, 0.001, 0.01, 0.1, 0.9]]
ax1.contour(x_vecs[index], v_vecs[index], p_xv[index].reshape((num_vals, num_vals)), colors="C1", levels=levels)

ax2 = fig.add_subplot(gs[:4, 4])
ax2.plot(p_v[index], v_vecs[index])
ax2.set_xticks([])

ax3 = fig.add_subplot(gs[4, :4])
ax3.plot(x_vecs[index], p_x[index])
ax3.set_yticks([])

ax1.sharey(ax2)
ax1.sharex(ax3)
ax1.set_xlim(x_vecs[index,0], x_vecs[index,-1])
ax1.set_ylim(v_vecs[index,0], v_vecs[index,-1])
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)

ax1.set_ylabel("$\dot{x}$")
ax3.set_xlabel("$x$")
ax3.set_ylabel("$p(x)$")
ax2.set_xlabel("$p(\dot{x})$")
ax1.set_title("$p(\dot{x}, x)$")

fig.savefig("p_xv_after_upcrossing.png")
plt.show()

## 

# fig, ax = plt.subplots(3, 3)
# for i in range(9):
#     joint = p_xv[index].reshape((num_vals, num_vals))
# 
#     ax.flat[i].plot(
# 
# soln_fn_xv_args = jax.vmap(ou_soln_xv_after_upcrossing_arguments, in_axes=(0, None, None, None, None, None, None))
# soln_fn_xv_test = jax.vmap(ou_soln_xv_after_upcrossing, in_axes=(0, None, None, None, None, None, None))
# quad, q, t2 = soln_fn_xv_args(xv_grids[index], 
#                        mu_xy[upcrossing_ind_1],
#                        s_xv[upcrossing_ind_1],
#                        b[index],
#                        b_dot[index],
#                        t_vec[index],
#                        p)
# 
# a = soln_fn_xv_test(xv_grids[index], 
#                        mu_xy[upcrossing_ind_1],
#                        s_xv[upcrossing_ind_1],
#                        b[index],
#                        b_dot[index],
#                        t_vec[index],
#                        p)
# 
# t2 = arg * jnp.exp(arg ** 2) * np.pi ** 0.5 * jsp.special.erfc(arg)
# 
#  
# 
# p_v_b = p_v_b / p_b[:,None]
# 
# p_v_b = jnp.nan_to_num(p_v_b)
# p_b = jnp.nan_to_num(p_b)
# def compute_mu_sigma(v, p_v):
#     e = jnp.trapz(p_v * v, x=v)
#     s = jnp.trapz(p_v * (v - e) ** 2, x=v)
#     return e, s
# 
# compute_mu_sigma = jax.vmap(compute_mu_sigma, in_axes=(None, 0))
# mu, sigma = compute_mu_sigma(v_vec, p_v_b)
# 
# nfunc = jax.vmap(integral_f1_xdot, in_axes=(None, 0, 0))
# 
# n1 = nfunc(0, mu, sigma) * p_b
# 
#  
# 
# soln_fn_v_b = jax.vmap(ou_soln_xv_after_upcrossing_arguments, in_axes=(0, None, None, None, None, None, None))
# soln_fn_v_b = jax.vmap(soln_fn_v_b, in_axes=(None, None, None, None, None, 0, None))
# 
# vmin, vmax = xv[:,:,0].min(), xv[:,:,0].max()
# vdiff = vmax - vmin
# vmin, vmax = vmin - 0.5 * vdiff, vmax + 0.5 * vdiff
# 
# v_vec = np.linspace(vmin, vmax, 10001)
# z_arr = np.stack((v_vec, np.zeros_like(v_vec) + p.threshold), axis=1)
# 
# quad, q = soln_fn_v_b(z_arr,
#                       mu_xy[upcrossing_ind_1],
#                       s_xv[upcrossing_ind_1],
#                       p.threshold,
#                       0.,
#                       t_vec,
#                       p)
# 
# def func1(quad):
#     return np.exp(quad)
# 
# def func2(q, quad):
#     return -q * np.exp(q ** 2 + quad) * np.pi ** 0.5 * erfc(q)
# 
# def func3(q, quad):
#     return -q * np.exp(q ** 2 + quad)
# 
# 
# index = 100
# 
# f1 = func1(quad[index])
# f2 = func2(q[index], quad[index])
# f3 = func3(q[index], quad[index])
# 
# fvals = f1 + f2
# fvals /= fvals.sum()
# a = np.arange(len(fvals))
# e = np.sum(a * fvals)
# s = np.sum((a - e) ** 2 * fvals)
# n = norm.pdf(a, loc=e, scale=np.sqrt(s))
# 
# # plt.figure()
# # plt.plot(v_vec, fvals)
# # plt.plot(v_vec, n, '--')
# 
# plt.figure()
# plt.plot(v_vec, f1)
# plt.plot(v_vec, f2)
# plt.plot(v_vec, f1 + f2)
# 
# plt.show()
# 
# t1 = np.trapz(f1, x=v_vec)
# t2 = np.trapz(f2, x=v_vec)
# print(f"t1: {t1}, t2: {t2}, t1/t2: {t1 / t2}")

##

