import sys
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import Simulator, SimulationParameters
import matplotlib.pyplot as plt
import numpy as np


p = SimulationParameters(threshold = 0.02, num_procs=100000)


n1s = []
n2sims = []
n2comps = []
for seed in range(100,150):
    sim = Simulator(params=p, crossing_times=[3.0, 4.5, 7.5], seed=seed)
    n1sim = sim.simulate_n1()
    n1comp = sim.compute_n1()
    n2sim = sim.simulate_n2()
    n2comp = sim.compute_n2()     
    n2sims.append(n2sim)
    n2comps.append(n2comp)
n2sims = np.array(n2sims)
n2comps = np.array(n2comps)
