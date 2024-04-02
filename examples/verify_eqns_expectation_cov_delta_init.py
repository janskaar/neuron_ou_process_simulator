import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.stats import multivariate_normal
import sys, os, h5py
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, MomentsSimulator, MembranePotentialSimulator, ParticleSimulator
from neurosim.n_functions import compute_n1, pdf_b, xv_to_xy, xy_to_xv, integral_f1_xdot, compute_mu_var_v_upcrossing

p = SimulationParameters(threshold=0.02, dt=0.01, I_e = 0., num_procs=100000)

t = 6.
num_steps = int(t / p.dt)

u_0 = 0.
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)

sim1 = MomentsSimulator(mu_0, s_0, p)
sim1.simulate(t)



