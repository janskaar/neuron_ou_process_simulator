import numpy as np
import os, h5py
import matplotlib.pyplot as plt

savefile = os.path.join("save", "plot_n2_N1_N2.h5")
num_steps = 1001

N1 = np.zeros(num_steps, dtype=np.float64)
N2 = np.zeros((num_steps, num_steps), dtype=np.float64)
with h5py.File(savefile, "r") as f:
    keys = sorted(list(f.keys()))

    for key in keys:
        print(f"PROCESSING KEY {key}", flush=True)
        N1 += f[key]["N1"][()]
        N2 += f[key]["N2"][()]

np.save(os.path.join("save", "plot_n2_N1.npy"), N1)
np.save(os.path.join("save", "plot_n2_N2.npy"), N2)
