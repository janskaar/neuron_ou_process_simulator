import sys
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import Simulator, SimulationParameters
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz

p = SimulationParameters(threshold = 0.02, num_procs=100000, dt=0.1)

#sim = Simulator(params=p, crossing_times=[30.0])
#n1sim, n1s = sim.simulate_n1(return_sims=True)

num_steps = 501
mus = []
sigmas = []
n1cs = np.zeros(num_steps, dtype=np.float64)
n2cs = np.zeros((num_steps, num_steps), dtype=np.float64)
for i in range(num_steps):
    sim = Simulator(params=p, crossing_times=[i*p.dt])
    n1 = sim.compute_n1()
    n1 = n1[0]
    if np.isnan(n1):
        n1 = 0.
    n1cs[i] = n1

# for i in range(num_steps):
#     print(i, end="\r")
#     for j in range(i+1, num_steps, 1):
#         sim = Simulator(params=p, crossing_times=[i*p.dt, j*p.dt])
#         n2 = sim.compute_n2()
#         n2 = n2[0]
#         if np.isnan(n2):
#             n2 = 0.
#         n2cs[i,j] = n2



