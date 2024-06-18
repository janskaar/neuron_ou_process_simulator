import numpy as np
import matplotlib.pyplot as plt
import sys, os, h5py, time
sys.path.append("/mnt/users/janskaar/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, ParticleSimulator

RANK = int(os.environ["SLURM_PROCID"])
savedir = "save_fix_x"
savefile = os.path.join(savedir, "plot_n2_N1_N2.h5")

p = SimulationParameters(threshold=0.02, dt=0.01, I_e = 0., num_procs=100000)

t = 6.

u_0 = 0.
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)

if RANK == 0:
    with h5py.File(savefile, "w") as f:
        pass
        
z_0 = np.zeros((p.num_procs, 2), dtype=np.float64)

save_xy_ind = 300
num_per_rank = 100
start = RANK * num_per_rank + 1
for seed in range(start, start + num_per_rank, 1):
    np.random.seed(seed)
    sim = ParticleSimulator(z_0.copy(), 0., p, fix_x_threshold=True) 
    sim.simulate(t)
    sim.compute_N1_N2()

    for i in range(100):
        try:
            with h5py.File(savefile, "r+") as f:
                grp = f.create_group(str(seed))
                grp.create_dataset("N1", data=sim.N1)
                grp.create_dataset("N2", data=sim.N2)
            break
        except BlockingIOError:
            time.sleep(0.5) 

    print(f"{seed} complete", flush=True)
print("COMPLETE")

