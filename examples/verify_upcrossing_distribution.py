import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.stats import multivariate_normal
import sys, os, h5py
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, MomentsSimulator, MembranePotentialSimulator, ParticleSimulator
from neurosim.n_functions import compute_n1, pdf_b, xy_to_xv, integral_f1_xdot, compute_p_y_upcrossing_constant_b, compute_E_v_v2_upcrossing_numerical

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

z0 = np.zeros((p.num_procs, 2), dtype=np.float64)
pSim = ParticleSimulator(z0,
                         u_0,
                         p)

pSim.simulate(t)

time_ind = 440
inds = pSim.upcrossings[time_ind]

mu_upcrossing, s_upcrossing = compute_p_y_upcrossing_constant_b(0.02, s_xv[time_ind], n1[time_ind])
mu_numerical, s_numerical = compute_E_v_v2_upcrossing_numerical(0.02, 0., [0., 0.], s_xv[time_ind], n1[time_ind])

z_upcrossing = pSim.z[time_ind][pSim.upcrossings[time_ind]] 
v_upcrossing = -z_upcrossing[:,1] / p.tau_x + z_upcrossing[:,0] / p.C

print(f"Expected mean upcrossing: {mu_upcrossing.squeeze()}")
print(f"Sample mean upcrossing: {v_upcrossing.mean()}")

print(f"Expected var upcrossing: {s_upcrossing.squeeze()}")
print(f"Sample var upcrossing: {v_upcrossing.var()}")


