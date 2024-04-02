import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from functools import partial
from scipy.integrate import cumtrapz
from scipy.stats import multivariate_normal, norm
from scipy.special import erfc, erf
import sys, os, time
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, MomentsSimulator, MembranePotentialSimulator, ParticleSimulator
from neurosim.n_functions import compute_n1, xv_to_xy, xy_to_xv, integral_f1_xdot
from neurosim.n_functions import ou_soln_xv_after_upcrossing, ou_soln_marginal_x_after_upcrossing, ou_soln_marginal_v_after_upcrossing
from neurosim.n_functions import conditional_bivariate_gaussian

phi = np.arctan(1080 / 1920)
sz = (14 * np.cos(phi), 14 * np.sin(phi))

## ====================
p = SimulationParameters(threshold=0.01, dt=0.01, I_e = 0.1, num_procs=10000)

t = 20.
t_vec = np.arange(0, t, p.dt)

mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)
msim = MomentsSimulator(mu_0, s_0, p)
msim.simulate(t)
mu_xy = msim.mu
s_xy = msim.s
mu_xv, s_xv = xy_to_xv(mu_xy, s_xy, p)

membraneSim = MembranePotentialSimulator(0., p)
membraneSim.simulate(t)
b = membraneSim.b.copy()
b_dot = membraneSim.b_dot.copy()



num_procs = 160000000
n1sim = np.load("save/plot_n2_N1.npy")
n2sim = np.load("save/plot_n2_N2.npy")
fptsim = np.load("save/plot_n2_FPT.npy")
z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
sim = ParticleSimulator(z_0, 0., p)
sim.simulate(t)
xv = np.zeros_like(sim.z)
xv[...,1] = sim.z[...,1]
xv[...,0] = -xv[...,1] / p.tau_x + sim.z[...,0] / p.C

vmin, vmax = xv[:,:,0].min(), xv[:,:,0].max()
vdiff = vmax - vmin
vmin, vmax = vmin - 0.2 * vdiff, vmax + 0.2 * vdiff

xmin, xmax = xv[:,:,0].min(), xv[:,:,0].max()
xdiff = xmax - xmin
xmin, xmax = xmin - 0.2 * xdiff, xmax + 0.2 * xdiff

soln_fn_xv = partial(ou_soln_xv_after_upcrossing, p=p)
soln_fn_xv = jax.vmap(soln_fn_xv, in_axes=(0, None, None, None, None, None))
# soln_fn_xv = jax.vmap(soln_fn_xv, in_axes=(None, None, None, None, None, 0))
soln_fn_xv = jax.jit(soln_fn_xv)

soln_fn_x = partial(ou_soln_marginal_x_after_upcrossing, p=p)
#soln_fn_x = jax.vmap(soln_fn_x, in_axes=(None, None, None, None, None, 0))
soln_fn_x = jax.jit(soln_fn_x)

soln_fn_v = partial(ou_soln_marginal_v_after_upcrossing, p=p)
#soln_fn_v = jax.vmap(soln_fn_v, in_axes=(None, None, None, None, None, 0))
soln_fn_v = jax.jit(soln_fn_v)



## 

gs = GridSpec(4,4)
index = 200
mu_v_x, s_v_x = conditional_bivariate_gaussian(b[index], mu_xv[index], s_xv[index])

vmin, vmax = mu_v_x - 5 * np.sqrt(s_v_x), mu_v_x + 5 * np.sqrt(s_v_x)
xmin, xmax = -0.03, 0.03

v_vec = np.linspace(vmin, vmax, 1001)
x_vec = np.linspace(xmin, xmax, 1001)
vv, xx = np.meshgrid(v_vec, x_vec, indexing="ij")
z_arr = np.stack((vv, xx), axis=-1).reshape((-1, 2))

for t in np.linspace(0.01, 0.2, 11):
    p_x = soln_fn_x(x_vec,
                     mu_xv[index],
                     s_xv[index],
                     b[index],
                     b_dot[index],
                     t).squeeze()
 
    p_xv = soln_fn_xv(z_arr,
                      mu_xv[index],
                      s_xv[index],
                      b[index],
                      b_dot[index],
                      t).squeeze()
    
    fig = plt.figure()
    xax = fig.add_subplot(gs[3,:3])
    vax = fig.add_subplot(gs[:3,3])
#     xax.plot(
#     plt.pcolormesh(vv, xx, p_xv.reshape((len(v_vec), len(x_vec))))
#     plt.show()
    break

## 

# soln_fn_v_b = partial(ou_soln_xv_after_upcrossing, p=p)
# soln_fn_v_b = jax.vmap(soln_fn_v_b, in_axes=(0, None, None, None, None, None))
# soln_fn_v_b = jax.vmap(soln_fn_v_b, in_axes=(None, None, None, None, None, 0))
# soln_fn_v_b = jax.jit(soln_fn_v_b)
# 
# soln_fn_x = partial(ou_soln_marginal_x_after_upcrossing, p=p)
# soln_fn_x = jax.vmap(soln_fn_x, in_axes=(None, None, None, None, None, 0))
# soln_fn_x = jax.jit(soln_fn_x)
# 
# soln_fn_v = partial(ou_soln_marginal_v_after_upcrossing, p=p)
# soln_fn_v = jax.vmap(soln_fn_v, in_axes=(None, None, None, None, None, 0))
# soln_fn_v = jax.jit(soln_fn_v)
# 
# vmin, vmax = xv[:,:,0].min(), xv[:,:,0].max()
# vdiff = vmax - vmin
# vmin, vmax = vmin - 0.2 * vdiff, vmax + 0.2 * vdiff
# 
# v_vec = np.linspace(vmin, vmax, 101)
# z_arr = np.stack((v_vec, np.zeros_like(v_vec) + p.threshold), axis=1)
# 
# n1 = compute_n1(p.threshold, 0, mu_xv, s_xv)
# 
# def compute_mu_sigma(v, p_v):
#     e = jnp.trapz(p_v * v, x=v)
#     s = jnp.trapz(p_v * (v - e) ** 2, x=v)
#     return e, s
# 
# compute_mu_sigma = jax.vmap(compute_mu_sigma, in_axes=(None, 0))
# compute_mu_sigma = jax.jit(compute_mu_sigma)

# nfunc = jax.vmap(integral_f1_xdot, in_axes=(None, 0, 0))
# 
# n2_arr = np.zeros((len(t_vec), len(t_vec)), dtype=np.float64)
# for i, t in enumerate(t_vec):
#     print(f"Index {i}", end="\r")
#     last_index = len(t_vec) if i == 0 else -i
# 
#     p_v_b = soln_fn_v_b(z_arr,
#                   mu_xy[i],
#                   s_xv[i],
#                   p.threshold,
#                   0.,
#                   t_vec).squeeze()
# 
#     p_b = soln_fn_x(p.threshold,
#                     mu_xy[i],
#                     s_xv[i],
#                     p.threshold,
#                     0.,
#                     t_vec).squeeze()
#     
#     p_v_b = p_v_b / p_b[:,None]
#     
#     p_v_b = jnp.nan_to_num(p_v_b)
#     p_b = jnp.nan_to_num(p_b)
# 
#     mu, sigma = compute_mu_sigma(v_vec, p_v_b)
#     
#     n2 = nfunc(0, mu, sigma) * p_b * n1[i]
# 
#     n2_arr[i,i:] = n2[:last_index]
# 
# n2 = n2_arr * num_procs * p.dt ** 2


