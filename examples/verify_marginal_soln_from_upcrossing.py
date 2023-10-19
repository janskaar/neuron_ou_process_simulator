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
from neurosim.n_functions import compute_n1, pdf_b, xv_to_xy, xy_to_xv, integral_f1_xdot
from neurosim.n_functions import ou_soln_v_upcrossing_v_delta_x, compute_p_v_upcrossing, conditional_bivariate_gaussian, gaussian_pdf
from neurosim.n_functions import ou_soln_x_upcrossing_v_delta_x
from neurosim.n_functions import ou_soln_marginal_x_upcrossing_v_delta_x
from neurosim.n_functions import ou_soln_xv_upcrossing_v_delta_x
from neurosim.n_functions import ou_soln_xv_integrand
from neurosim.n_functions import ou_soln_upcrossing_alpha_beta, ou_soln_upcrossing_S

phi = np.arctan(1080 / 1920)
sz = (14 * np.cos(phi), 14 * np.sin(phi))

sim_ind = 300

# simulate expectation / covariance for first upcrossing
t1_sim = 10.
p = SimulationParameters(threshold=0.01, dt=0.01, I_e = 0., num_procs=100000)
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)
msim = MomentsSimulator(mu_0, s_0, p)
msim.simulate(t1_sim)
mu_xy = msim.mu
s_xy = msim.s
mu_xv, s_xv = xy_to_xv(mu_xy, s_xy, p)
upcrossing_ind = 900

v = np.linspace(0., 0.1, 1001)
p_v = compute_p_v_upcrossing(v, p.threshold, 0, mu_xv[sim_ind], s_xv[sim_ind]).squeeze()
p_v_integral = np.trapz(p_v, x=v)
print(f"p(v) INTEGRAL: {p_v_integral}")
p_v /= p_v_integral

z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
v_0 = np.random.choice(v, p=p_v * (v[1] - v[0]), size=p.num_procs, replace=True)
y_0 = (v_0 + p.threshold / p.tau_x) * p.C
z_0[:,0] = y_0
z_0[:,1] = p.threshold
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
xmins, xmaxs = [0.0099, 0.0095, 0.002], [0.0105, 0.012, 0.03]

f1 = integral_f1_xdot(p.threshold, 0, mu_xv, s_xv)
mu_0, s_0 = conditional_bivariate_gaussian(p.threshold, mu_xv[sim_ind], s_xv[sim_ind])
f_0 = f1[sim_ind]
b_dot_0 = 0
b_0 = p.threshold

fig, ax = plt.subplots(ncols=3, nrows=2)

t_plots = [0.02, 0.1, 1.0]

for i in range(3):
    vs = np.linspace(vmins[i], vmaxs[i], 1001)
    xs = np.linspace(xmins[i], xmaxs[i], 1001)
    t_plot = t_plots[i]
    ind_plot = int(t_plot / p.dt)
    
    ou_soln_v = ou_soln_v_upcrossing_v_delta_x(vs, mu_0, s_0, f_0, b_0, b_dot_0, t_plot, p)
    ou_soln_x_old = ou_soln_x_upcrossing_v_delta_x(xs, mu_0, s_0, 1, b_0, b_dot_0, t_plot, p)
    
    ou_soln_x = ou_soln_marginal_x_upcrossing_v_delta_x(xs, mu_xv[sim_ind], s_xv[sim_ind], p.threshold, 0, t_plot, p)
    ax[0,i].plot(vs, ou_soln_v, label="analytical")
    _, ymax = ax[0,i].get_ylim()
    ax[0,i].set_ylim(0, ymax)
    twax = ax[0,i].twinx()
    twax.hist(xv[ind_plot,:,0], density=True, bins=vs, histtype="step", color="C1", label="simulation")
    twax.set_yticks([])    
    ax[0,i].set_xlabel("$\dot{x}$")

    ax[1,i].plot(xs, ou_soln_x)
#    ax[1,i].plot(xs, ou_soln_x_old)
    _, ymax = ax[1,i].get_ylim()
    ax[1,i].set_ylim(0, ymax)
    twax = ax[1,i].twinx()
    twax.hist(xv[ind_plot,:,1], density=True, bins=xs, histtype="step", color="C1")
    twax.set_yticks([])    
    a, b = twax.get_ylim()
    twax.set_ylim(a, b * 0.9)
    ax[1,i].set_xlabel("$x$")
    ax[0,i].set_title(f"{t_plot:.1f} ms")

ax[0,0].plot(0,0, label="simulation")
ax[0,0].set_ylabel("pdf")
ax[1,0].set_ylabel("pdf")
ax[0,0].legend()

plt.show()

p_x = ou_soln_marginal_x_upcrossing_v_delta_x(xs, mu_xv[sim_ind], s_xv[sim_ind], p.threshold, 0, 1.0, p)
plt.plot(p_x)
plt.twinx()
plt.plot(ou_soln_x, "C1")
plt.show()





