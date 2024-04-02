import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.stats import multivariate_normal, norm
import sys, os, h5py
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, MomentsSimulator, MembranePotentialSimulator, ParticleSimulator
from neurosim.n_functions import compute_n1, pdf_b, xy_to_xv, compute_p_y_crossing, integral_f1_xdot, compute_E_y_upcrossing_constant_b, compute_E_y2_upcrossing_constant_b, compute_p_y_upcrossing_constant_b

p = SimulationParameters(threshold=0.02, dt=0.01, I_e = 0., num_procs=100000, sigma_noise=50.)

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
mu = sim.mu
s = sim.s

mu_xv, s_xv = xy_to_xv(mu, s, p)

n1 = compute_n1(b[0], b_dot[0], mu_xv, s_xv)

sim_ind = 35 

E_y = compute_E_y_upcrossing_constant_b(b[sim_ind], s_xv[sim_ind], n1[sim_ind])
E_y2 = compute_E_y2_upcrossing_constant_b(b[sim_ind], s_xv[sim_ind], n1[sim_ind])
p_b = pdf_b(0.02, mu_xv[sim_ind,1], s_xv[sim_ind,2])

num_samples = 1000000
cov_xv = np.array([[s_xv[sim_ind,0], s_xv[sim_ind,1]],
                   [s_xv[sim_ind,1], s_xv[sim_ind,2]]])


samples = multivariate_normal.rvs(mean=mu_xv[sim_ind], cov=cov_xv, size=num_samples)
inds = (samples[:,0] * p.dt + samples[:,1] >= 0.02) & (samples[:,1] < 0.02) & (samples[:,0] > 0.)
num = inds.sum()
print(f"n1: {n1[sim_ind]}")
print(f"integral: {num / num_samples / p.dt}")
print()
print(f"E_y: {E_y / n1[sim_ind]}")
print(f"E_y_samples: {samples[inds,0].mean()}")
print()
print(f"E_y2: {E_y2 / n1[sim_ind]}")
print(f"E_y2_samples: {(samples[inds,0]**2).mean()}")


mu_v_b, s_v_b = compute_p_y_crossing(b[sim_ind], mu_xv[sim_ind], s_xv[sim_ind]) 

eps = 0.00005
line_inds = (samples[:,1] < 0.02+eps) & (samples[:,1] > 0.02-eps) & (samples[:,0] > 0.)


z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)
pSim = ParticleSimulator(z_0,
                         u_0,
                         p)

pSim.simulate(t)

z_crossing = pSim.z[sim_ind][pSim.upcrossings[sim_ind] | pSim.downcrossings[sim_ind]]


z_upcrossing = pSim.z[sim_ind][pSim.upcrossings[sim_ind]] 
v_upcrossing = -z_upcrossing[:,1] / p.tau_x + z_upcrossing[:,0] / p.C
print()
print(f"V_upcrossing mean: {v_upcrossing.mean()}")
print(f"V_upcrossing mean v^2: {(v_upcrossing**2).mean()}")


# integral3 = compute_E_y_upcrossing_constant_b(b[sim_ind], s_xv[sim_ind])
# 
# mu_v_b, s_v_b = compute_p_y_crossing(0.02, mu_xv[sim_ind], s_xv[sim_ind])
# p_b = pdf_b(0.02, mu_xv[sim_ind,1], s_xv[sim_ind,2])
# 
# vline = np.linspace(mu_v_b - 10 * s_v_b ** 0.5, mu_v_b + 10 * s_v_b ** 0.5, 1001)
# int_inds = vline >= 0.
# 
# pdf = norm.pdf(vline, loc=mu_v_b, scale=np.sqrt(s_v_b))
# m = np.trapz(vline[int_inds] ** 2 * pdf[int_inds], dx=vline[1] - vline[0])




