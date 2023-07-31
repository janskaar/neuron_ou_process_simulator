import numpy as np
import os, h5py
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

savefile = os.path.join("save", "plot_n2_upcrossings.h5")
num_steps = 1001

n1 = np.zeros(num_steps, dtype=np.float64)
n2 = np.zeros((num_steps, num_steps), dtype=np.float64)
with h5py.File(savefile, "r") as f:
    keys = sorted(list(f.keys()))

    keys_per_rank = len(keys) // size
    remainder = len(keys) % size
    local_key_indices = [keys_per_rank*rank + i for i in range(keys_per_rank)]
    local_keys = [keys[i] for i in local_key_indices]
    if rank < remainder:
        local_key_indices.append(keys_per_rank*size+rank)
        local_keys.append(keys[-(rank+1)])


    for key in local_keys:
        print(f"RANK {rank} PROCESSING KEY {key}", flush=True)
        upcrossings = f[key][()]

        n1 += upcrossings.sum(1).astype(np.float64)

        for i in range(num_steps):
            for j in range(i+1, num_steps, 1):
                n2[i,j] += float((upcrossings[i,:] & upcrossings[j,:]).sum())

if rank == 0:
    n2sum = np.zeros_like(n2)
    n1sum = np.zeros_like(n1)
else:
    n2sum = None
    n1sum = None


comm.Barrier()
comm.Reduce(
    [n2, MPI.DOUBLE],
    [n2sum, MPI.DOUBLE],
    op = MPI.SUM,
    root = 0
)


comm.Barrier()
comm.Reduce(
    [n1, MPI.DOUBLE],
    [n1sum, MPI.DOUBLE],
    op = MPI.SUM,
    root = 0
)


if rank == 0:
    np.save(os.path.join("save", "plot_n2_n1.npy"), n1sum)
    np.save(os.path.join("save", "plot_n2_n2.npy"), n2sum)
