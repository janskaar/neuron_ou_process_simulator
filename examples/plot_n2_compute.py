import numpy as np
import os, h5py
import matplotlib.pyplot as plt

savefile = os.path.join("save", "plot_n2_upcrossings.h5")

num_steps = 1001
n1 = np.zeros(num_steps, dtype=np.float64)
n2 = np.zeros((num_steps, num_steps), dtype=np.float64)
with h5py.File(savefile, "r") as f:
    keys = list(f.keys())
    for key in keys:
        upcrossings = f[key][()]
        n1 += upcrossings.sum(1).astype(np.float64)
        for i in range(num_steps):
            for j in range(i+1, num_steps, 1):
                n2[i,j] += float((upcrossings[:,i] & upcrossings[:,j]).sum())

np.save(os.path.join("save", "plot_n2_n1.npy"), n1)
np.save(os.path.join("save", "plot_n2_n2.npy"), n2)
