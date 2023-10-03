import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jax
import jax.numpy as jnp
from scipy.integrate import cumtrapz
from scipy.stats import multivariate_normal
from scipy.linalg import expm
import sys, os
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, MomentsSimulator, MembranePotentialSimulator, ParticleSimulator
from neurosim.n_functions import compute_n1, pdf_b, xv_to_xy, xy_to_xv, integral_f1_xdot, compute_mu_var_v_upcrossing
from neurosim.n_functions import ou_soln_v_upcrossing_v_delta_x, compute_p_v_upcrossing, conditional_bivariate_gaussian
from neurosim.n_functions import ou_soln_x_upcrossing_v_delta_x
from neurosim.n_functions import ou_soln_xv_upcrossing_v_delta_x
from neurosim.n_functions import ou_soln_upcrossing_alpha_beta, ou_soln_upcrossing_S



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
# f1 = integral_f1_xdot(b, b_dot, mu_xv1, s_xv1)
# n1 = compute_n1(b, b_dot, mu_xv1, s_xv1)
# 
# z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
# sim1 = ParticleSimulator(z_0, 0., p)
# sim1.simulate(t)
# xv1 = np.zeros_like(sim1.z)
# xv1[...,1] = sim1.z[...,1]
# xv1[...,0] = -xv1[...,1] / p.tau_x + sim1.z[...,0] / p.C


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
# v_vec = np.linspace(v.min(), v.max(), 101)
# 
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



# Verify alpha / beta and S

p = SimulationParameters(threshold=0.01, dt=0.01, I_e = 0., num_procs=10000)

t = 100.
num_steps = int(t / p.dt)
t_vec = np.arange(0, t+p.dt, p.dt)


def E_y(t, y_0, p):
    return y_0 * np.exp(-t / p.tau_y)

def E_x(t, x_0, y_0, p):
    exp_tau_x = np.exp(-t / p.tau_x)
    exp_tau_y = np.exp(-t / p.tau_y)
    delta_e = exp_tau_x - exp_tau_y
    denom1 = (1. / p.tau_y - 1. / p.tau_x)
    return y_0 * delta_e / (p.C * denom1) + x_0 * exp_tau_x
 
z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
z_0[:,1] = p.threshold
z_0[:,0] = 1.3
sim = ParticleSimulator(z_0, 0., p)
sim.simulate(t)
xv = np.zeros_like(sim.z)
xv[...,1] = sim.z[...,1]
xv[...,0] = -sim.z[...,1] / p.tau_x + sim.z[...,0] / p.C
v_0 = - p.threshold / p.tau_x + 1.3 / p.C
alpha, beta = ou_soln_upcrossing_alpha_beta(t_vec, p)
S_func = jax.vmap(ou_soln_upcrossing_S, in_axes=(0, None))
S = S_func(t_vec, p)


gs = GridSpec(2,6)
phi = np.arctan(1080 / 1920)
sz = (14 * np.cos(phi), 14 * np.sin(phi)),  
fig = plt.figure()
fig.set_size_inches(*sz)

ax1 = fig.add_subplot(gs[0, :3])
ax1.plot(xv[...,0].mean(1))
ax1.plot(alpha[0] * v_0 + beta[0] * p.threshold)
ax1.set_title("$E[\dot{x}]$")

ax2 = fig.add_subplot(gs[0, 3:])
ax2.plot(xv[...,1].mean(1))
ax2.plot(alpha[1] * v_0 + beta[1] * p.threshold)
ax2.set_title("$E[x]$")

ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(xv[...,0].var(1))
ax3.plot(S[:,0,0])
ax3.set_title("$Cov(\dot{x}, \dot{x})$")

ax4 = fig.add_subplot(gs[1, 2:4])
cov = np.mean((xv[...,0] - xv[...,0].mean(1, keepdims=True)) * (xv[...,1] - xv[...,1].mean(1, keepdims=True)), axis=1)
ax4.plot(cov)
ax4.plot(S[:,0,1])
ax4.set_title("$Cov(\dot{x}, x)$")

ax5 = fig.add_subplot(gs[1, 4:])
ax5.plot(xv[...,1].var(1))
ax5.plot(S[:,1,1])
ax5.set_title("$Cov(x, x)$")

gs.update(wspace=1.)

plt.show()


# Verify soln after upcrossing

# t = 10.
# p = SimulationParameters(threshold=0.01, dt=0.01, I_e = 0., num_procs=10000)
# mu_0 = np.zeros(2, dtype=np.float64)
# s_0 = np.zeros(3, dtype=np.float64)
# msim = MomentsSimulator(mu_0, s_0, p)
# msim.simulate(5.)
# mu_xy = msim.mu
# s_xy = msim.s
# mu_xv, s_xv = xy_to_xv(mu_xy, s_xy, p)
# upcrossing_ind = 300
# 
# v_vec = np.linspace(0., 0.1, 1001)
# p_v = compute_p_v_upcrossing(v_vec, p.threshold, 0., mu_xv[upcrossing_ind], s_xv[upcrossing_ind]).squeeze()
# p_v_integral = np.trapz(p_v, x=v_vec)
# print(f"p(v) INTEGRAL: {p_v_integral}")
# p_v /= p_v_integral
# 
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

