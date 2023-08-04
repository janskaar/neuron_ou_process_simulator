import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.stats import multivariate_normal
import sys, os, h5py
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, MomentsSimulator, MembranePotentialSimulator, ParticleSimulator
from neurosim.n_functions import compute_n1, pdf_b, xy_to_xv, compute_p_y_crossing, integral_f1_xdot

p = SimulationParameters(threshold=0.02, dt=0.01, I_e = 0., num_procs=100000)

t = 5.
num_steps = int(t / p.dt)

u_0 = 0.
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)

uSim = MembranePotentialSimulator(u_0, p)
uSim.simulate(t)
u = uSim.u
b = uSim.b
b_dot = uSim.b_dot

sim = MomentsSimulator(mu_0, s_0, p)
sim.simulate(t)
mu_n1 = sim.mu
s_n1 = sim.s

mu_xv, s_xv = xy_to_xv(mu_n1, s_n1, p)

n1 = compute_n1(b[0], b_dot[0], mu_xv, s_xv)
n1s = np.zeros((num_steps+1, num_steps+1), dtype=np.float64)
mu_ys = []
s_ys = []
for i in range(1, num_steps):
    mu_y, s_y = compute_p_y_crossing(b[i], mu_n1[i], s_n1[i])
    mu_ys.append(mu_y)
    s_ys.append(s_y)
    mu = np.array([mu_y, b[i]])
    s = np.array([s_y, 0., 0.])
    sim = MomentsSimulator(mu, s, p)
    sim.simulate(t-i*p.dt)
    mu_xv, s_xv = xy_to_xv(sim.mu, sim.s, p)
    # n1 conditioned on spiking at step i
    n1_c = compute_n1(b[i], b_dot[i], mu_xv, s_xv)
    n1s[i,i:] = n1_c

sim_ind = 300

mu = np.array([mu_ys[sim_ind], b[sim_ind]])
s = np.array([s_ys[sim_ind], 0., 0.])

sim = MomentsSimulator(mu, s, p)
sim.simulate(10. - sim_ind * p.dt)
mu_xv, s_xv = xy_to_xv(sim.mu, sim.s, p)
# n1 conditioned on spiking at step i
n1_c = compute_n1(b[i], b_dot[i], mu_xv, s_xv)

z0 = np.zeros((p.num_procs, 2), dtype=np.float64)
z0[:,1] = b[sim_ind]
z0[:,0] = np.random.randn(p.num_procs) * np.sqrt(s_ys[sim_ind]) + mu_ys[sim_ind]
pSim = ParticleSimulator(z0,
                         u[sim_ind],
                         p)

pSim.simulate(t - sim_ind * p.dt)
pSim.compute_N1_N2()

n2s = n1s * n1[None,:]

R = 2 * n1s / n1[None,:] - 1
R = np.nan_to_num(R, nan=-1.)
integrand = n1s - n1[None,:]
integrand[np.tril_indices(integrand.shape[0])] = 0.

num_sim_procs = 100000 * 16 * 10
n1sim = np.load(os.path.join("save", "plot_n2_N1.npy"))
n2sim = np.load(os.path.join("save", "plot_n2_N2.npy"))

xs = np.load(os.path.join("save", "xs.npy"))
ys = np.load(os.path.join("save", "ys.npy"))

xs_t_minus_1 = np.load(os.path.join("save", "xs_t_minus_1.npy"))
ys_t_minus_1 = np.load(os.path.join("save", "ys_t_minus_1.npy"))

xs_crossing = np.load(os.path.join("save", "xs_crossing.npy"))
ys_crossing = np.load(os.path.join("save", "ys_crossing.npy"))

mu, s = compute_p_y_crossing(b[sim_ind], mu_n1[sim_ind], s_n1[sim_ind])

integral_f1_xdot(b[sim_ind], b_dot[sim_ind], mu_n1[sim_ind,0

z1 = np.stack((ys_t_minus_1[:], xs_t_minus_1[:]), axis=1)
p1 = SimulationParameters(threshold=0.02, dt=0.001, I_e = 0., num_procs=len(z1))

sim1 = ParticleSimulator(z1,
                         u[sim_ind],
                         p1)

sim1.simulate(0.01)

crossings = sim1.upcrossings | sim1.downcrossings





# z1 = np.stack((ys, xs), axis=1)
# z1 = np.concatenate([z1]*25, axis=0)
# #z1[:,1] = b[sim_ind]
# p1 = SimulationParameters(threshold=0.02, dt=0.01, I_e = 0., num_procs=len(z1))
# sim1 = ParticleSimulator(z1,
#                          u[sim_ind],
#                          p1)
# 
# sim1.simulate(t - sim_ind * p.dt)
# sim1.compute_N1_N2()
# 
# 
# p2 = SimulationParameters(threshold=0.02, dt=0.01, I_e = 0., num_procs=len(z1))
# 
# z2 = np.zeros((len(z1), 2), dtype=np.float64)
# z2[:,1] = b[sim_ind]
# z2[:,0] = np.random.randn(len(z2)) * z1[:,0].std() + z1[:,0].mean()  #* np.sqrt(s_ys[sim_ind]) + mu_ys[sim_ind]
# sim2 = ParticleSimulator(z2,
#                          u[sim_ind],
#                          p2)
# 
# sim2.simulate(t - sim_ind * p.dt)
# sim2.compute_N1_N2()
# 
# 
# plt.plot(sim1.upcrossings.sum(1))
# plt.plot(sim2.upcrossings.sum(1))
# plt.show()

