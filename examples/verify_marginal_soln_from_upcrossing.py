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
from neurosim.n_functions import compute_n1, pdf_b, xv_to_xy, xy_to_xv, integral_f1_xdot_gaussian
from neurosim.n_functions import compute_p_v_upcrossing, conditional_bivariate_gaussian, gaussian_pdf
from neurosim.n_functions import ou_soln_marginal_x_after_upcrossing
from neurosim.n_functions import ou_soln_marginal_v_after_upcrossing
from neurosim.n_functions import ou_soln_xv_after_upcrossing
from neurosim.n_functions import ou_soln_xv_integrand
from neurosim.n_functions import ou_soln_upcrossing_alpha_beta, ou_soln_upcrossing_S

phi = np.arctan(1080 / 1920)
sz = (14 * np.cos(phi), 14 * np.sin(phi))

sim_ind = 300

# simulate expectation / covariance for first upcrossing
t1_sim = 10.
p = SimulationParameters(threshold=0.01, dt=0.01, I_e = 0.1, num_procs=100000)
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)
msim = MomentsSimulator(mu_0, s_0, p)
msim.simulate(t1_sim)
mu_xy = msim.mu
s_xy = msim.s
mu_xv, s_xv = xy_to_xv(mu_xy, s_xy, p)
upcrossing_ind = 900

usim = MembranePotentialSimulator(0., p)
usim.simulate(t1_sim)
b = usim.b
b_dot = usim.b_dot

v = np.linspace(0., 0.1, 10001)
p_v = compute_p_v_upcrossing(v, b[sim_ind], b_dot[sim_ind], mu_xv[sim_ind], s_xv[sim_ind]).squeeze()

z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
v_0 = np.random.choice(v, p=p_v / p_v.sum(), size=p.num_procs, replace=True)
y_0 = (v_0 + b[sim_ind] / p.tau_x) * p.C
z_0[:,0] = y_0
z_0[:,1] = b[sim_ind]
sim = ParticleSimulator(z_0, 0., p)
sim.simulate(1.)
xv = np.zeros_like(sim.z)
xv[...,1] = sim.z[...,1]
xv[...,0] = -xv[...,1] / p.tau_x + sim.z[...,0] / p.C

def E_y(y_0, t, p):
    return y_0 * np.exp(-t / p.tau_y)

def var_y(t, p):
    return (p.sigma2_noise * p.tau_y / 2.) * (1 - np.exp(-2 * t / p.tau_y))

vmins, vmaxs = [-0.025, -0.025, -0.025], [0.025, 0.025, 0.025]
xmins, xmaxs = [0.0099, 0.0095, 0.002], [0.0105, 0.05, 0.04]
vmin, vmax = -0.025, 0.025
xmin, xmax = 0.005, 0.04

f1 = integral_f1_xdot_gaussian(b[sim_ind], b_dot[sim_ind], mu_xv, s_xv)
mu_0, s_0 = conditional_bivariate_gaussian(b[sim_ind], mu_xv[sim_ind], s_xv[sim_ind])
f_0 = f1[sim_ind]
b_dot_0 = b_dot[sim_ind]
b_0 = b[sim_ind]

fig, ax = plt.subplots(ncols=3, nrows=2)
fig.set_size_inches(sz)

t_plots = [0.02, 0.1, 1.0]

for i in range(3):
    vs = np.linspace(vmin, vmax, 10001)
    xs = np.linspace(xmin, xmax, 10001)
    t_plot = t_plots[i]
    ind_plot = int(t_plot / p.dt)
    
    ou_soln_v = ou_soln_marginal_v_after_upcrossing(vs, mu_xv[sim_ind], s_xv[sim_ind], b[sim_ind], b_dot[sim_ind], t_plot, p)
    ou_soln_x = ou_soln_marginal_x_after_upcrossing(xs, mu_xv[sim_ind], s_xv[sim_ind], b[sim_ind], b_dot[sim_ind], t_plot, p)

    a_, b_ = xv[ind_plot,:,0].min(), xv[ind_plot,:,0].max()
    a_ -= np.abs(a_) * 0.01
    b_ += np.abs(b_) * 0.01
    bins = np.linspace(a_, b_, 101)
    ax[0,i].hist(xv[ind_plot,:,0], density=True, bins=bins, histtype="step", color="C1", label="simulation")
    ax[0,i].plot(vs, ou_soln_v, label="analytical", linestyle="--")
    ax[0,i].set_xlabel("$\dot{x}$")

    a_, b_ = xv[ind_plot,:,1].min(), xv[ind_plot,:,1].max()
    a_ -= np.abs(a_) * 0.01
    b_ += np.abs(b_) * 0.01
    bins = np.linspace(a_, b_, 101)
    ax[1,i].hist(xv[ind_plot,:,1], density=True, bins=bins, histtype="step", color="C1")
    ax[1,i].plot(xs, ou_soln_x, linestyle="--")
    ax[1,i].set_xlabel("$x$")
    ax[0,i].set_title(f"{t_plot:.1f} ms")

ax[0,0].set_ylabel("pdf")
ax[1,0].set_ylabel("pdf")
ax[0,0].legend()

plt.show()


