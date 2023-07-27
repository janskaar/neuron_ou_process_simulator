import numpy as np
import matplotlib.pyplot as plt
import sys, os, h5py, time
sys.path.append("/p/project/cdeep/skaar1/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, ParticleSimulator

RANK = int(os.environ["SLURM_PROCID"])
print(f"RANK {RANK}\n", flush=True)
savedir = "save"
savefile = os.path.join(savedir, "plot_n2_sims.h5")


p = SimulationParameters(threshold=0.02, dt=0.1, I_e = 0., num_procs=100000)

t = 100.

u_0 = 0.
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)

if RANK == 0:
    with h5py.File(savefile, "w") as f:
        pass
        
z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)

num_per_rank = 5
start = RANK * num_per_rank + 1
for seed in range(start, start + num_per_rank, 1):
    print(f"{seed} complete \n", flush=True)
    np.random.seed(seed)
    sim = ParticleSimulator(z_0.copy(), u_0, p) 
    sim.simulate(t)
    for i in range(100):
        try:
            with h5py.File(savefile, "r+") as f:
                f.create_dataset(str(seed), data=sim.upcrossings)
                print("SAVING {seed}\n", flush=True)
            break
        except BlockingIOError:
            print(f"{seed} FILE LOCKED, WAITING", flush=True)
            time.sleep(0.5) 
