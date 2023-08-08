import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, MomentsSimulator, MembranePotentialSimulator, ParticleSimulator
from neurosim.n_functions import compute_n1, pdf_b, xy_to_xv, f1_schwalger


p1 = SimulationParameters(threshold=0.02, dt=0.01, I_e = 0., num_procs=100000, sigma_noise = 1.5, C = 1.)
D = p1.tau_y * p1.sigma2_noise / p1.tau_x ** 2
s_y = D / p1.tau_y
p1.sigma_noise = np.sqrt(2 * D) / p1.tau_y

u_0 = 0.
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)
s_0[0] = s_y


uSim = MembranePotentialSimulator(u_0, p1)
uSim.simulate(20.)


schwalgerSim = MomentsSimulator(mu_0, s_0, p1)
schwalgerSim.simulate(20.)
f1 = f1_schwalger(uSim.b, uSim.b_dot, schwalgerSim.mu, schwalgerSim.s, p1)


# p2 = SimulationParameters(threshold=0.02, dt=0.01, I_e = 0., num_procs=100000, sigma_noise = 1.5, C = 1.)
p2 = p1
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)
s_0[0] = p2.sigma2_noise * p2.tau_y / 2.

momentsSim = MomentsSimulator(mu_0, s_0, p2)

momentsSim.simulate(20.)


mu, s = xy_to_xv(momentsSim.mu, momentsSim.s, p2)
n1 = compute_n1(uSim.b, uSim.b_dot, mu, s)


plt.plot(n1)
plt.plot(f1)
plt.show()


