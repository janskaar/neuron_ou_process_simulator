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
from neurosim.n_functions import ou_soln_upcrossing_alpha_beta, ou_soln_upcrossing_S

phi = np.arctan(1080 / 1920)
sz = (14 * np.cos(phi), 14 * np.sin(phi))


# u_0 = 0.
# mu_0 = np.zeros(2, dtype=np.float64)
# s_0 = np.zeros(3, dtype=np.float64)
# 
# usim1 = MembranePotentialSimulator(u_0, p)
# usim1.simulate(t)
# u = usim1.u
# b = usim1.b
# b_dot = usim1.b_dot
# 
# msim1 = MomentsSimulator(mu_0, s_0, p)
# msim1.simulate(t)
# mu_xy1 = msim1.mu
# s_xy1 = msim1.s
# mu_xv1, s_xv1 = xy_to_xv(mu_xy1, s_xy1, p)
# f1 = integral_f1_xdot_gaussian(b, b_dot, mu_xv1, s_xv1)
# n1 = compute_n1(b, b_dot, mu_xv1, s_xv1)
# 
# z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
# sim1 = ParticleSimulator(z_0, 0., p)
# sim1.simulate(t)
# xv1 = np.zeros_like(sim1.z)
# xv1[...,1] = sim1.z[...,1]
# xv1[...,0] = -xv1[...,1] / p.tau_x + sim1.z[...,0] / p.C


## ====================
# Verify upcrossing distribution p(v|upcrossing)

# p = SimulationParameters(threshold=0.01, dt=0.01, I_e = 0., num_procs=500000)
# z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
# z_0[:,1] = 0.
# z_0[:,0] = 0.
# sim = ParticleSimulator(z_0, 0., p)
# sim.simulate(5.)
# i = sim.upcrossings.sum(1).argmax()
# z = sim.z[i,sim.upcrossings[i]]
# v = -z[:,1] / p.tau_x + z[:,0] / p.C
# v_vec = np.linspace(v.min(), v.max(), 1001)
# 
# mu_0 = np.zeros(2, dtype=np.float64)
# s_0 = np.zeros(3, dtype=np.float64)
# msim = MomentsSimulator(mu_0, s_0, p)
# msim.simulate(5.)
# mu_xy = msim.mu
# s_xy = msim.s
# mu_xv, s_xv = xy_to_xv(mu_xy, s_xy, p)
# 
# p_v = compute_p_v_upcrossing(v_vec, p.threshold, 0, mu_xv[i], s_xv[i]).squeeze()
# 
# fig, ax = plt.subplots(1)
# ax.hist(v, bins=50, density=True);
# ax.plot(v_vec, p_v)
# plt.show()


## ====================
# Verify alpha / beta and S

# p = SimulationParameters(threshold=0.01, dt=0.01, I_e = 0., num_procs=10000)
# 
# t = 10.
# num_steps = int(t / p.dt)
# t_vec = np.arange(0, t+p.dt, p.dt)
# 
# y_0 = 1.3
# 
# def E_y(t, y_0, p):
#     return y_0 * np.exp(-t / p.tau_y)
# 
# def E_x(t, x_0, y_0, p):
#     exp_tau_x = np.exp(-t / p.tau_x)
#     exp_tau_y = np.exp(-t / p.tau_y)
#     delta_e = exp_tau_x - exp_tau_y
#     denom1 = (1. / p.tau_y - 1. / p.tau_x)
#     return y_0 * delta_e / (p.C * denom1) + x_0 * exp_tau_x
#  
# z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
# z_0[:,1] = p.threshold
# z_0[:,0] = y_0
# sim = ParticleSimulator(z_0, 0., p)
# sim.simulate(t)
# xv = np.zeros_like(sim.z)
# xv[...,1] = sim.z[...,1]
# xv[...,0] = -sim.z[...,1] / p.tau_x + sim.z[...,0] / p.C
# v_0 = - p.threshold / p.tau_x + y_0 / p.C
# alpha, beta = ou_soln_upcrossing_alpha_beta(t_vec, p)
# S_func = jax.vmap(ou_soln_upcrossing_S, in_axes=(0, None))
# S = S_func(t_vec, p)
# 
# gs = GridSpec(2,6)
# phi = np.arctan(1080 / 1920)
# sz = (14 * np.cos(phi), 14 * np.sin(phi)),  
# fig = plt.figure()
# fig.set_size_inches(*sz)
# 
# ax1 = fig.add_subplot(gs[0, :3])
# ax1.plot(xv[...,0].mean(1))
# ax1.plot(alpha[0] * v_0 + beta[0] * p.threshold)
# ax1.set_title("$E[\dot{x}]$")
# 
# ax2 = fig.add_subplot(gs[0, 3:])
# ax2.plot(xv[...,1].mean(1))
# ax2.plot(alpha[1] * v_0 + beta[1] * p.threshold)
# ax2.set_title("$E[x]$")
# 
# ax3 = fig.add_subplot(gs[1, :2])
# ax3.plot(xv[...,0].var(1))
# ax3.plot(S[:,0,0])
# ax3.set_title("$Cov(\dot{x}, \dot{x})$")
# 
# ax4 = fig.add_subplot(gs[1, 2:4])
# cov = np.mean((xv[...,0] - xv[...,0].mean(1, keepdims=True)) * (xv[...,1] - xv[...,1].mean(1, keepdims=True)), axis=1)
# ax4.plot(cov)
# ax4.plot(S[:,0,1])
# ax4.set_title("$Cov(\dot{x}, x)$")
# 
# ax5 = fig.add_subplot(gs[1, 4:])
# ax5.plot(xv[...,1].var(1))
# ax5.plot(S[:,1,1])
# ax5.set_title("$Cov(x, x)$")
# 
# gs.update(wspace=1.)
# 
# plt.show()

## ====================
# Verify integrand

# # simulate expectation / covariance for first upcrossing
# t1_sim = 10.
# p = SimulationParameters(threshold=0.01, dt=0.01, I_e = 0.1, num_procs=10000)
# mu_0 = np.zeros(2, dtype=np.float64)
# s_0 = np.zeros(3, dtype=np.float64)
# msim = MomentsSimulator(mu_0, s_0, p)
# msim.simulate(t1_sim)
# mu_xy = msim.mu
# s_xy = msim.s
# mu_xv, s_xv = xy_to_xv(mu_xy, s_xy, p)
# 
# usim = MembranePotentialSimulator(0., p)
# usim.simulate(t1_sim)
# b = usim.b
# b_dot = usim.b_dot
# 
# upcrossing_ind = 900
# 
# t = 10.
# num_steps = int(t / p.dt)
# t_vec = np.arange(0, t+p.dt, p.dt)
# 
# y_0 = 1.3
# x_0 = 0.01 
# 
# 
# z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
# z_0[:,0] = y_0
# z_0[:,1] = x_0
# np.random.seed(1234)
# sim = ParticleSimulator(z_0, 0., p)
# sim.simulate(t)
# xv = np.zeros_like(sim.z)
# xv[...,1] = sim.z[...,1]
# xv[...,0] = -sim.z[...,1] / p.tau_x + sim.z[...,0] / p.C
# v_0 = - x_0 / p.tau_x + y_0 / p.C
# 
# plot_ind = 50
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
# soln_fn = jax.vmap(ou_soln_xv_integrand, in_axes=(0, None, None, None, None, None))
# f = soln_fn(z_arr,
#             v_0,
#             x_0,
#             b_dot[upcrossing_ind],
#             t_vec[plot_ind],
#             p)
# 
# f = f.reshape((101, 101))
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(xv2[:,0], xv2[:,1], s=1.)
# ax.contour(vv, xx, f)
# plt.show()

## ====================

# Verify soln after upcrossing

# # simulate expectation / covariance for first upcrossing
# t1_sim = 10.
# p = SimulationParameters(threshold=0.01, dt=0.01, I_e = 0., num_procs=10000)
# mu_0 = np.zeros(2, dtype=np.float64)
# s_0 = np.zeros(3, dtype=np.float64)
# msim = MomentsSimulator(mu_0, s_0, p)
# msim.simulate(t1_sim)
# mu_xy = msim.mu
# s_xy = msim.s
# mu_xv, s_xv = xy_to_xv(mu_xy, s_xy, p)
# upcrossing_ind = 900
# 
# # compute p(v) at upcrossing
# v_vec = np.linspace(0., 0.1, 1001)
# p_v = compute_p_v_upcrossing(v_vec, p.threshold, 0., mu_xv[upcrossing_ind], s_xv[upcrossing_ind]).squeeze()
# p_v_integral = np.trapz(p_v, x=v_vec)
# print(f"p(v) INTEGRAL: {p_v_integral}")
# p_v /= p_v_integral
# 
# 
# # simulate with initial conditions of upcrossing
# t = 10.
# t_vec = np.arange(0, t+p.dt, p.dt)
# z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
# v_0 = np.random.choice(v_vec, p=p_v * (v_vec[1] - v_vec[0]), size=p.num_procs, replace=True)
# y_0 = (v_0 + p.threshold / p.tau_x) * p.C
# z_0[:,0] = y_0
# z_0[:,1] = p.threshold
# sim = ParticleSimulator(z_0, 0., p)
# sim.simulate(t)
# xv = np.zeros_like(sim.z)
# xv[...,1] = sim.z[...,1]
# xv[...,0] = -xv[...,1] / p.tau_x + sim.z[...,0] / p.C
# 
# soln_fn = jax.vmap(ou_soln_xv_after_upcrossing, in_axes=(0, None, None, None, None, None, None))
# 
# plot_ind = 10
# 
# zmin, zmax = xv[plot_ind,:,0].min(), xv[plot_ind,:,0].max()
# xmin, xmax = xv[plot_ind,:,1].min(), xv[plot_ind,:,1].max()
# 
# v_vec = np.linspace(zmin, zmax, 101)
# x_vec = np.linspace(xmin, xmax, 101)
# vv, xx = np.meshgrid(v_vec, x_vec)
# z_arr = np.stack((vv, xx), axis=-1).reshape((-1, 2))
# 
# f = soln_fn(z_arr,
#             mu_xy[upcrossing_ind],
#             s_xv[upcrossing_ind],
#             p.threshold,
#             0.,
#             t_vec[plot_ind],
#             p)
# f = np.array(f)
# f[np.isnan(f)] = 0
# 
# f = f.reshape((101, 101))
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(xv[plot_ind,:,0], xv[plot_ind,:,1], s=1.)
# ax.contour(vv, xx, f)
# plt.show()
# 
# integral = np.sum(f) * (v_vec[1] - v_vec[0]) * (x_vec[1] - x_vec[0])
# print(f"Integral: {integral}")

## ====================

# Verify second upcrossing probability
# NOTE: The plots look wrong at the very beginning of the simulation,
# but this is a discretization artefact. Increasing the resolution 10-fold,
# the same artefact is still present for the first ~20 time steps, but correct
# after.

# simulate expectation / covariance for first upcrossing

# t1_sim = 10.
# p = SimulationParameters(threshold=0.001, dt=0.01, I_e = 0., num_procs=10000)
# mu_0 = np.zeros(2, dtype=np.float64)
# s_0 = np.zeros(3, dtype=np.float64)
# msim = MomentsSimulator(mu_0, s_0, p)
# msim.simulate(t1_sim)
# mu_xy = msim.mu
# s_xy = msim.s
# mu_xv, s_xv = xy_to_xv(mu_xy, s_xy, p)
# upcrossing_ind_1 = 900
# 
# # compute p(v) at upcrossing
# v_vec = np.linspace(0., 0.1, 10001)
# p_v = compute_p_v_upcrossing(v_vec, p.threshold, 0., mu_xv[upcrossing_ind_1], s_xv[upcrossing_ind_1]).squeeze()
# p_v_integral = np.trapz(p_v, x=v_vec)
# print(f"p(v) INTEGRAL: {p_v_integral}")
# p_v /= p_v_integral
# 
# # simulate with initial conditions of upcrossing
# t = 1.
# t_vec = np.arange(0, t+p.dt, p.dt)
# z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
# v_0 = np.random.choice(v_vec, p=p_v * (v_vec[1] - v_vec[0]), size=p.num_procs, replace=True)
# y_0 = (v_0 + p.threshold / p.tau_x) * p.C
# z_0[:,0] = y_0
# z_0[:,1] = p.threshold
# sim = ParticleSimulator(z_0, 0., p)
# sim.simulate(t)
# xv = np.zeros_like(sim.z)
# xv[...,1] = sim.z[...,1]
# xv[...,0] = -xv[...,1] / p.tau_x + sim.z[...,0] / p.C
# 
# soln_fn_v_b = jax.vmap(ou_soln_xv_after_upcrossing, in_axes=(0, None, None, None, None, None, None))
# soln_fn_v_b = jax.vmap(soln_fn_v_b, in_axes=(None, None, None, None, None, 0, None))
# 
# soln_fn_x = jax.vmap(ou_soln_marginal_x_after_upcrossing, in_axes=(None, None, None, None, None, 0, None))
# 
# soln_fn_v = jax.vmap(ou_soln_marginal_v_after_upcrossing, in_axes=(None, None, None, None, None, 0, None))
# 
# vmin, vmax = xv[:,:,0].min(), xv[:,:,0].max()
# vdiff = vmax - vmin
# vmin, vmax = vmin - 0.5 * vdiff, vmax + 0.5 * vdiff
# 
# v_vec = np.linspace(vmin, vmax, 10001)
# z_arr = np.stack((v_vec, np.zeros_like(v_vec) + p.threshold), axis=1)
# 
# p_v_b = soln_fn_v_b(z_arr,
#               mu_xy[upcrossing_ind_1],
#               s_xv[upcrossing_ind_1],
#               p.threshold,
#               0.,
#               t_vec,
#               p).squeeze()
# 
# p_b = soln_fn_x(p.threshold,
#                 mu_xy[upcrossing_ind_1],
#                 s_xv[upcrossing_ind_1],
#                 p.threshold,
#                 0.,
#                 t_vec,
#                 p).squeeze()
# 
# p_v = soln_fn_v(v_vec,
#                 mu_xy[upcrossing_ind_1],
#                 s_xv[upcrossing_ind_1],
#                 p.threshold,
#                 0.,
#                 t_vec,
#                 p).squeeze()
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
# plt.plot(sim.upcrossings.sum(1) / p.num_procs)
# plt.plot(n1 * p.dt)
# plt.show()
# 
# for i in range(10):
#     t_ind = i * 1
#     a = xv[t_ind,:,0]
#     plt.hist(a, bins=100, density=True)
#     plt.plot(v_vec, p_v[t_ind])
#     plt.show()
# 
# eps = 0.000001
# b_inds = (sim.z[...,1] > p.threshold - eps) & (sim.z[...,1] <= p.threshold)
# 
# for i in range(20):
#     t_ind = i 
#     a = xv[t_ind,b_inds[t_ind],0]
#     plt.hist(a, bins=100, density=True, histtype="step")
#     plt.plot(v_vec, p_v_b[t_ind])
#     plt.show()

## ##INVESTIGATE MSTERIOUS FUNCTION

# soln_fn = jax.vmap(ou_soln_xv_after_upcrossing, in_axes=(0, None, None, None, None, None, None))
# 
# plot_ind = 100
# 
# vmin, vmax = xv[plot_ind,:,0].min(), xv[plot_ind,:,0].max()
# vdiff = vmax - vmin
# vmin, vmax = vmin - 0.5 * vdiff, vmax + 0.5 * vdiff
# xmin, xmax = xv[plot_ind,:,1].min(), xv[plot_ind,:,1].max()
# xdiff = xmax - xmin
# xmin, xmax = xmin - 0.5 * xdiff, xmax + 0.5 * xdiff
# xmean = xv[plot_ind,:,1].mean()
# 
# v_vec = np.linspace(vmin, vmax, 1001)
# # x_vec = np.array([xmin, p.threshold, xmean])
# x_vec = np.linspace(xmin, xmax, 1001)
# vv, xx = np.meshgrid(v_vec, x_vec, indexing="ij")
# z_arr = np.stack((vv, xx), axis=-1).reshape((-1, 2))
# 
# f = soln_fn(z_arr,
#             mu_xy[upcrossing_ind_1],
#             s_xv[upcrossing_ind_1],
#             p.threshold,
#             0.,
#             t_vec[plot_ind],
#             p)
# 
# f = f.reshape((len(v_vec), len(x_vec)))
# 
# for i in range(0,400, 40):
#     fig = plt.figure()
#     fig.set_size_inches(sz)
#     a = f[:,i]
#     nan_a = np.asarray(a).copy()
#     nan_a[np.isnan(nan_a)] = 0
#     a_norm = nan_a / np.trapz(nan_a, x=v_vec)
# 
#     dx = v_vec[1] - v_vec[0]
# 
#     e = np.sum(a_norm * v_vec) * dx
#     s = np.sum(a_norm * (v_vec - e ) ** 2) * dx
#     n = norm.pdf(v_vec, loc=e, scale=s ** 0.5)
#     n = n / n.max() * np.nanmax(a) 
# 
#     plt.plot(v_vec, a, c="royalblue", label="Final dist.")
#     plt.plot(v_vec, n, c="black", label="Normal approx.", linestyle="--")
#     plt.legend()
# 
#     plt.show()
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
# vvec = np.linspace(-0.015, 0.01, 1001)
# xval = 0.01
# xvec = np.zeros_like(vvec) + xval
# z = np.stack((vvec, xvec), axis=1)
# 
# mu0 = mu_xy[upcrossing_ind_1]
# s0 = s_xv[upcrossing_ind_1]
# 
# def compute_q_d(z, b_0, b_dot_0, t, mu_0, s_0):
#     alpha, beta = ou_soln_upcrossing_alpha_beta(t, p)
#     beta *= p.threshold
#     mu_v_x, s_v_x = conditional_bivariate_gaussian(b_0, mu_0, s_0) 
#     S = ou_soln_upcrossing_S(t, p)
#     S_inv = jnp.linalg.solve(S, jnp.eye(2))
#     c0 = (z - beta).T.dot(S_inv).dot(z - beta) + mu_v_x ** 2 / s_v_x
#     c1 = 2 * alpha.T.dot(S_inv).dot(-z + beta) - 2 * mu_v_x / s_v_x
#     c2 = alpha.T.dot(S_inv).dot(alpha) + 1. / s_v_x
#     dvec = -0.5 * (c0 + b_dot_0 * (c1 + b_dot_0 * c2))
#     qvec = (c1 + 2 * b_dot_0 * c2) / (2 ** 1.5 * c2 ** 0.5)
#     return qvec, dvec
# 
# compute_q_d_vec = jax.vmap(compute_q_d, in_axes=(0,None,None,None,None,None))
# qs, ds = compute_q_d_vec(z, p.threshold, 0, 0.01, mu0, s0)
# 
# qvec = np.linspace(-15, 15, 1001)
# xvec = np.linspace(-250, 10, 1001)
# qq, xx = np.meshgrid(qvec, xvec, indexing="ij")
# grid = np.stack((qq, xx), axis=-1).reshape((-1, 2))
# 
# vals = []
# for (q_, x_) in grid:
#     vals.append(func2(q_, x_) + func1(x_))
# vals = np.array(vals).reshape((len(qvec), len(xvec)))

## 

# zvals = np.linspace(-5, 5, 1001)
# 
# qvals = 1 * zvals
# quadvals = -4 * zvals ** 2
# 
# def func1(quad):
#     return np.exp(quad)
# 
# def func2(q, quad):
#     return -q * np.exp(q ** 2 + quad) * np.pi ** 0.5 * erfc(q)
# 
# def func3(q):
#     return -q * np.exp(-q ** 2)
# 
# def func4(q):
#     return -q * np.exp(-q ** 2) * erfc(q)
# 
# f1 = func1(quadvals)
# f2 = func2(qvals, quadvals)
# f3 = func3(qvals)
# f4 = func4(qvals)
# 
# plt.plot(zvals, f1)
# plt.plot(zvals, f2)
# plt.plot(zvals, f1 + f2)
# plt.plot(zvals, f3)
# plt.plot(zvals, f4)
# plt.show()
# 
# 
# qq, xx = np.meshgrid(qvals, quadvals, indexing="ij")
# grid = np.stack((qq, xx), axis=-1).reshape((-1, 2))
# 
# fvals = func1(quadvals) + func2(qvals, quadvals)
# fvals /= fvals.sum()
# a = np.arange(len(fvals))
# e = np.sum(a * fvals)
# s = np.sum((a - e) ** 2 * fvals)
# n = norm.pdf(a, loc=e, scale=np.sqrt(s))
# 
# plt.figure()
# plt.plot(qvals, fvals)
# plt.plot(qvals, n, '--')
# 
# plt.figure()
# plt.pcolormesh(qq, xx, vals, vmin=-1, vmax=1)
# plt.scatter(qvals, quadvals, s=1.)
# # plt.xlim(qvec[0], qvec[-1])
# # plt.ylim(xvec[0], xvec[-1])
# plt.show()




## 

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(xv[plot_ind,:,0], xv[plot_ind,:,1], s=1.)
# ax.contour(vv, xx, f)
# plt.show()

# v_vec = np.linspace(-0.025, 0.025, 1001)
# z_arr = np.stack((v_vec, np.zeros_like(v_vec)+p.threshold), axis=-1)
# 
# f = soln_fn(z_arr,
#             mu_xy[upcrossing_ind],
#             s_xv[upcrossing_ind],
#             p.threshold,
#             0.,
#             t_vec[plot_ind],
#             p)
# 
# f = np.array(f)
# f[np.isnan(f)] = 0.
# f = f / np.trapz(f, x=v_vec)
# 
# e = np.trapz(f * v_vec, x=v_vec)
# s = np.trapz(f * (v_vec - e) ** 2, x=v_vec)
# g = norm.pdf(v_vec, loc=e, scale=s**0.5)
# 
# inds = (xv[plot_ind,:,1] < 0.01005) & (xv[plot_ind,:,1] >= 0.00995)
# 
# plt.plot(v_vec, f)
# plt.plot(v_vec, g)
# # plt.hist(xv[plot_ind,inds,0], density=True, bins=100)
# plt.show()

## ====================

# sim_ind = 300
# 
# 
# v = np.linspace(0., 0.1, 1001)
# p_v = compute_p_v_upcrossing(v, b[sim_ind], b_dot[sim_ind], mu_xv1[sim_ind], s_xv1[sim_ind]).squeeze()
# p_v_integral = np.trapz(p_v, x=v)
# print(f"p(v) INTEGRAL: {p_v_integral}")
# p_v /= p_v_integral
# 
# # xs_crossing = np.load(os.path.join("save", "xs_crossing.npy"))
# # ys_crossing = np.load(os.path.join("save", "ys_crossing.npy"))
# # vs_crossing = -xs_crossing / p.tau_x + ys_crossing / p.C
# 
# z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
# v_0 = np.random.choice(v, p=p_v * (v[1] - v[0]), size=p.num_procs, replace=True)
# y_0 = (v_0 + b[sim_ind] / p.tau_x) * p.C
# z_0[:,0] = y_0
# z_0[:,1] = p.threshold
# sim2 = ParticleSimulator(z_0, 0., p)
# sim2.simulate(t)
# xv2 = np.zeros_like(sim2.z)
# xv2[...,1] = sim2.z[...,1]
# xv2[...,0] = -xv2[...,1] / p.tau_x + sim2.z[...,0] / p.C
# 
# # z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
# # z_0[:,0] = 1.
# # z_0[:,1] = p.threshold
# # sim3 = ParticleSimulator(z_0, 0., p)
# # sim3.simulate(t)
# # 
# # mu_0 = np.array([1., p.threshold])
# # s_0 = np.array([0., 0., 0.])
# # msim3 = MomentsSimulator(mu_0, s_0, p)
# # msim3.simulate(t)
# # 
# # xv3 = np.zeros_like(sim3.z)
# # xv3[...,1] = sim3.z[...,1]
# # xv3[...,0] = -xv3[...,1] / p.tau_x + sim3.z[...,0] / p.C
# 
# # ts = np.arange(0, 200+p.dt, p.dt)
# # mu_xs = []
# # for i in ts:
# #     y_term, x_term = compute_mu_terms_t_upcrossing_y_delta_x(i, p)
# #     mu_xs.append(y_term + 0.02 * x_term)
# # mu_xs = np.array(mu_xs)
# # mu_ys = np.exp(-ts / p.tau_y)
# 
# 
# # A = np.array([[-1./p.tau_y, 0.],
# #               [1./p.C, -1./p.tau_x]])
# # expectation_prop = expm(A * 2.)
# # y_term, x_term = compute_mu_terms_t_upcrossing_y_delta_x(2., p)
#  
# 
# def E_y(y_0, t, p):
#     return y_0 * np.exp(-t / p.tau_y)
# 
# def var_y(t, p):
#     return (p.sigma2_noise * p.tau_y / 2.) * (1 - np.exp(-2 * t / p.tau_y))
# 
# 
# fig, ax = plt.subplots(2)
# ax[0].plot(sim1.z[...,0].mean(1))
# ax[0].plot(mu_xy1[:,0], '--')
# ax[1].plot(sim1.z[...,0].var(1))
# ax[1].plot(s_xy1[:,0], '--')
# plt.show()

# vs = np.linspace(-0.1, 0.1, 1001)
# xs = np.linspace(0., 0.025, 1001)
# 
# mu_0, s_0 = conditional_bivariate_gaussian(p.threshold, mu_xv1[sim_ind], s_xv1[sim_ind])
# f_0 = f1[sim_ind]
# b_dot_0 = b_dot[sim_ind]
# b_0 = b[sim_ind]
# 
# # f1, e1, e2, t1, t2 = ou_soln_upcrossing_v_delta_x(vs, mu_0, s_0, f_0, b_0, b_dot_0, 0.01, p)
# 
# fig, ax = plt.subplots(ncols=3, nrows=2)
# 
# t_plots = [0.02, 0.1, 1.0]
# 
# 
# for i in range(3):
#     t_plot = t_plots[i]
#     ind_plot = int(t_plot / p.dt)
#     
#     ou_soln_v = ou_soln_v_upcrossing_v_delta_x(vs, mu_0, s_0, f_0, b_0, b_dot_0, t_plot, p)
#     ou_soln_x = ou_soln_x_upcrossing_v_delta_x(xs, mu_0, s_0, f_0, b_0, b_dot_0, t_plot, p)
#     
#     ax[0,i].plot(vs, ou_soln_v, label="analytical")
#     _, ymax = ax[0,i].get_ylim()
#     ax[0,i].set_ylim(0, ymax)
#     twax = ax[0,i].twinx()
#     twax.hist(xv2[ind_plot,:,0], density=True, bins=vs, histtype="step", color="C1", label="simulation")
#     twax.set_yticks([])    
#     ax[0,i].set_xlabel("$x$")
# 
#     ax[1,i].plot(xs, ou_soln_x)
#     _, ymax = ax[1,i].get_ylim()
#     ax[1,i].set_ylim(0, ymax)
#     twax = ax[1,i].twinx()
#     twax.hist(xv2[ind_plot,:,1], density=True, bins=xs, histtype="step", color="C1")
#     twax.set_yticks([])    
# 
#     ax[1,i].set_xlabel("$\dot{x}$")
#     ax[0,i].set_title(f"{t_plot:.1f} ms")
# 
# ax[0,0].plot(0,0, label="simulation")
# ax[0,0].set_ylabel("pdf")
# ax[1,0].set_ylabel("pdf")
# ax[0,0].legend()
# 
# plt.show()

