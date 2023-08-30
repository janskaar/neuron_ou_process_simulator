import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.stats import multivariate_normal
from scipy.linalg import expm
import sys, os, h5py
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, MomentsSimulator, MembranePotentialSimulator, ParticleSimulator
from neurosim.n_functions import compute_n1, pdf_b, xv_to_xy, xy_to_xv, integral_f1_xdot, compute_mu_var_v_upcrossing
from neurosim.n_functions import ou_soln_v_upcrossing_v_delta_x, compute_p_v_upcrossing, conditional_bivariate_gaussian
from neurosim.n_functions import ou_soln_x_upcrossing_v_delta_x

p = SimulationParameters(threshold=0.01, dt=0.01, I_e = 0., num_procs=100000)

t = 6.
num_steps = int(t / p.dt)

u_0 = 0.
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)

usim1 = MembranePotentialSimulator(u_0, p)
usim1.simulate(t)
u = usim1.u
b = usim1.b
b_dot = usim1.b_dot

msim1 = MomentsSimulator(mu_0, s_0, p)
msim1.simulate(t)
mu_xy1 = msim1.mu
s_xy1 = msim1.s
mu_xv1, s_xv1 = xy_to_xv(mu_xy1, s_xy1, p)
f1 = integral_f1_xdot(b, b_dot, mu_xv1, s_xv1)
n1 = compute_n1(b, b_dot, mu_xv1, s_xv1)


z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
sim1 = ParticleSimulator(z_0, 0., p)
sim1.simulate(t)
xv1 = np.zeros_like(sim1.z)
xv1[...,1] = sim1.z[...,1]
xv1[...,0] = -xv1[...,1] / p.tau_x + sim1.z[...,0] / p.C


sim_ind = 300

v = np.linspace(0., 0.1, 1001)
p_v = compute_p_v_upcrossing(v, b[sim_ind], b_dot[sim_ind], mu_xv1[sim_ind], s_xv1[sim_ind]).squeeze()
p_v_integral = np.trapz(p_v, x=v)
print(f"p(v) INTEGRAL: {p_v_integral}")
p_v /= p_v_integral

# xs_crossing = np.load(os.path.join("save", "xs_crossing.npy"))
# ys_crossing = np.load(os.path.join("save", "ys_crossing.npy"))
# vs_crossing = -xs_crossing / p.tau_x + ys_crossing / p.C

z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
v_0 = np.random.choice(v, p=p_v * (v[1] - v[0]), size=p.num_procs, replace=True)
y_0 = (v_0 + b[sim_ind] / p.tau_x) * p.C
z_0[:,0] = y_0
z_0[:,1] = p.threshold
sim2 = ParticleSimulator(z_0, 0., p)
sim2.simulate(t)
xv2 = np.zeros_like(sim2.z)
xv2[...,1] = sim2.z[...,1]
xv2[...,0] = -xv2[...,1] / p.tau_x + sim2.z[...,0] / p.C

# z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
# z_0[:,0] = 1.
# z_0[:,1] = p.threshold
# sim3 = ParticleSimulator(z_0, 0., p)
# sim3.simulate(t)
# 
# mu_0 = np.array([1., p.threshold])
# s_0 = np.array([0., 0., 0.])
# msim3 = MomentsSimulator(mu_0, s_0, p)
# msim3.simulate(t)
# 
# xv3 = np.zeros_like(sim3.z)
# xv3[...,1] = sim3.z[...,1]
# xv3[...,0] = -xv3[...,1] / p.tau_x + sim3.z[...,0] / p.C

# ts = np.arange(0, 200+p.dt, p.dt)
# mu_xs = []
# for i in ts:
#     y_term, x_term = compute_mu_terms_t_upcrossing_y_delta_x(i, p)
#     mu_xs.append(y_term + 0.02 * x_term)
# mu_xs = np.array(mu_xs)
# mu_ys = np.exp(-ts / p.tau_y)


# A = np.array([[-1./p.tau_y, 0.],
#               [1./p.C, -1./p.tau_x]])
# expectation_prop = expm(A * 2.)
# y_term, x_term = compute_mu_terms_t_upcrossing_y_delta_x(2., p)
 

def E_y(y_0, t, p):
    return y_0 * np.exp(-t / p.tau_y)


def var_y(t, p):
    return (p.sigma2_noise * p.tau_y / 2.) * (1 - np.exp(-2 * t / p.tau_y))


fig, ax = plt.subplots(2)
ax[0].plot(sim1.z[...,0].mean(1))
ax[0].plot(mu_xy1[:,0], '--')
ax[1].plot(sim1.z[...,0].var(1))
ax[1].plot(s_xy1[:,0], '--')
plt.show()

vs = np.linspace(-0.1, 0.1, 1001)
xs = np.linspace(-0.1, 0.1, 1001)

mu_0, s_0 = conditional_bivariate_gaussian(p.threshold, mu_xv1[sim_ind], s_xv1[sim_ind])
f_0 = f1[sim_ind]
b_dot_0 = b_dot[sim_ind]
b_0 = b[sim_ind]

# f1, e1, e2, t1, t2 = ou_soln_upcrossing_v_delta_x(vs, mu_0, s_0, f_0, b_0, b_dot_0, 0.01, p)

t_plot = 0.1
ind_plot = int(t_plot / p.dt)

ou_soln_v = ou_soln_v_upcrossing_v_delta_x(vs, mu_0, s_0, f_0, b_0, b_dot_0, t_plot, p)
ou_soln_x = ou_soln_x_upcrossing_v_delta_x(xs, mu_0, s_0, f_0, b_0, b_dot_0, t_plot, p)


fig, ax = plt.subplots(2)

ax[0].plot(vs, ou_soln_v)
_, ymax = ax[0].get_ylim()
ax[0].set_ylim(0, ymax)
twax = ax[0].twinx()
twax.hist(xv2[ind_plot,:,0], density=True, bins=vs, histtype="step", color="C1")



ax[1].plot(xs, ou_soln_x)
_, ymax = ax[1].get_ylim()
ax[1].set_ylim(0, ymax)
twax = ax[1].twinx()
twax.hist(xv2[ind_plot,:,1], density=True, bins=vs, histtype="step", color="C1")

plt.show()


# plt.hist(vs_crossing, bins=1001, density=True)
# plt.plot(v, p_v[0])
# plt.show()

