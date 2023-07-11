import sys
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import Simulator, SimulationParameters
import matplotlib.pyplot as plt
import numpy as np


p = SimulationParameters(threshold = 0.02)

n1s = []
n2s = []
for seed in range(100,110):
    sim = Simulator(params=p, crossing_times=[3.0, 8.0], seed=seed)
    
    sim.simulate(10.)
    # sim.compute_n() 
    # plt.plot(sim.n * sim.p.num_procs * sim.p.dt)
    # plt.plot(sim.upcrossings.sum(1))
    # plt.show()
     
    n1s.append((sim.upcrossings[30].sum(), sim.upcrossings[80].sum()))
    n2s.append((sim.upcrossings[30] & sim.upcrossings[80]).sum())
    print("")
    print(np.exp(log_n))
    print((sim.upcrossings[30] & sim.upcrossings[80]).sum())
    print("") 



