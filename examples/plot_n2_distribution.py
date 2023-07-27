import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import Simulator, SimulationParameters

filepath = os.path.join("save", "n1_n2_sims.csv")




p = SimulationParameters(threshold = 0.02, num_procs=100000, dt=0.01)
sim = Simulator(params=p, crossing_times=[3.0, 4.5, 7.5])

n1comp = np.array(sim.compute_n1())
n2comp = np.array(sim.compute_n2())

nsim = pd.read_csv(filepath)
keys = nsim.keys()

fig, ax = plt.subplots(ncols=3, nrows=2)
for i, key in enumerate(keys):
    xmin = nsim[key].min()
    xmax = nsim[key].max()
    bins = np.arange(xmin, xmax+2, 1) - 0.1
    ax.flat[i].hist(nsim[key], bins=bins, density=True)
    ax.flat[i].set_title(key)

for i in range(3):
    ymax = ax[0,i].get_ylim()[1]
    ax[0,i].vlines(n1comp[i] * p.num_procs * p.dt, 0, ymax, color="red")
    ymax = ax[1,i].get_ylim()[1]
    ax[1,i].vlines(n2comp[i] * p.num_procs * p.dt**2, 0, ymax, color="red")
    
plt.show()
