import numpy as np
import matplotlib.pyplot as plt
import sys, os, h5py
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, ParticleSimulator

RANK = int(os.environ["SLURM_PROCID"])
NUM_PER_PROCESS = 100

savedir = "save"
savefile = os.path.join(savedir, "plot_n2_sims.h5")


p = SimulationParameters(threshold=0.02, dt=0.1, I_e = 0., num_procs=100000)

t = 100.

u_0 = 0.
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)

if RANK == 0:
    with h5py.File(savefile, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        fcntl.flock(f, fcntl.LOCK_UN)
        
z_0 = np.zeros((p.num_procs, 3), dtype=np.float64)

num_per_rank = 500
start = RANK * num_per_rank + 1
for seed in range(start, start + num_per_rank, 1):
    np.random.seed(seed)
    sim = ParticleSimulator(z_0.copy(), u_0, p) 
    sim.simulate(t)
    with h5py.File(savefile, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.create_dataset(str(seed), data=sim.upcrossings)
        fcntl.flock(f, fcntl.LOCK_UN)
 
