import fcntl
import sys, os, time
sys.path.append("/mnt/users/janskaar/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import Simulator, SimulationParameters
import matplotlib.pyplot as plt
import numpy as np

RANK = int(os.environ["SLURM_PROCID"])
print(RANK)

savedir = "save"
savefile = os.path.join(savedir, "n1_n2_sims.csv")

if RANK == 0:
    with open(savefile, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.writelines("n1_1,n1_2,n1_3,n2_1,n2_2,n2_3\n")
        fcntl.flock(f, fcntl.LOCK_UN)

p = SimulationParameters(threshold = 0.02, num_procs=100000, dt=0.01)

num_per_rank = 100
start = RANK * num_per_rank + 1
for seed in range(start, start + num_per_rank, 1):
    sim = Simulator(params=p, crossing_times=[3.0, 4.5, 7.5], seed=seed)
    n1sim = sim.simulate_n1()
    n1comp = sim.compute_n1()
    n2sim = sim.simulate_n2()
    n2comp = sim.compute_n2()     
    with open(savefile, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.writelines(f"{n1sim[0]},{n1sim[1]},{n1sim[2]},{n2sim[0]},{n2sim[1]},{n2sim[2]}\n")
        fcntl.flock(f, fcntl.LOCK_UN)

