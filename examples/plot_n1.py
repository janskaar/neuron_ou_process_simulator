import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, MomentsSimulator, MembranePotentialSimulator, ParticleSimulator
from neurosim.n_functions import compute_n1, pdf_b, xy_to_xv, f1_schwalger


p = SimulationParameters(threshold=0.02, dt=0.01, I_e = 0., num_procs=100000)
u_0 = 0.
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)

momentsSim = MomentsSimulator(mu_0, s_0, p)
uSim = MembranePotentialSimulator(u_0, p)
pSim = ParticleSimulator(np.zeros((p.num_procs, 2), dtype=np.float64),
                         u_0,
                         p)
pSim.simulate(20.)
uSim.simulate(20.)
momentsSim.simulate(20.)

mu, s = xy_to_xv(momentsSim.mu, momentsSim.s, p)
n1 = compute_n1(uSim.b, uSim.b_dot, mu, s)
f1 = f1_schwalger(uSim.b, uSim.b_dot, momentsSim.mu, momentsSim.s, p)


plt.plot(pSim.upcrossings.sum(1))
plt.plot(n1 * p.num_procs * p.dt)
plt.show()






