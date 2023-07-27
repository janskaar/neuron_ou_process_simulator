import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import sys
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, MomentsSimulator, MembranePotentialSimulator, ParticleSimulator
from neurosim.n_functions import compute_n1, pdf_b, xy_to_xv, compute_p_y_crossing

p = SimulationParameters(threshold=0.02, dt=0.1, I_e = 0., num_procs=10000)

t = 100.
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
for i in range(1, num_steps):
    mu, s = compute_p_y_crossing(b[i], mu_n1[i], s_n1[i])
    sim = MomentsSimulator(mu, s, p)
    sim.simulate(t-i*p.dt)
    mu_xv, s_xv = xy_to_xv(sim.mu, sim.s, p)
    # n1 conditioned on spiking at step i
    n1_c = compute_n1(b[i], b_dot[i], mu_xv, s_xv)
    n1s[i,i:] = n1_c



n2s = n1s * n1[None,:]
R = 2 * n1s / n1[None,:] - 1
R = np.nan_to_num(R, nan=-1.)
integrand = n1s - n1[None,:]
integrand[np.tril_indices(integrand.shape[0])] = 0.

pSim = ParticleSimulator(np.zeros((p.num_procs, 2), dtype=np.float64),
                         u_0,
                         p)

pSim.simulate(t)

