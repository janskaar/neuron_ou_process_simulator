"""
Numerical verification of formulas for the expectation and covariance from 
the OU-process given and arbitrary initial expectation and covariance

Implementation is based on equations [6-13] in accompanying note.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys, os, h5py
from scipy.linalg import expm
from scipy.stats import multivariate_normal
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, MomentsSimulator, MembranePotentialSimulator, ParticleSimulator
from neurosim.n_functions import compute_n1, pdf_b, xv_to_xy, xy_to_xv, integral_f1_xdot, compute_mu_var_v_upcrossing

p = SimulationParameters(threshold=0.02, dt=0.01, I_e = 0., num_procs=100000)

t = 10.
num_steps = int(t / p.dt)

def compute_expectation(m0, p, t):
    if m0.shape == (2,):
        m0 = m0[:,None]
    elif m0.shape == (1,2):
        m0 = m0.T

    A = np.array([[1./p.tau_y, 0.],
                  [-1./p.C, 1./p.tau_x]])
    exp_minus_At = expm(-A * t)
    return exp_minus_At.dot(m0)

def compute_cov(S0, p, t):
    A = np.array([[1./p.tau_y, 0.],
                  [-1./p.C, 1./p.tau_x]])
    exp_minus_At = expm(-A * t)

    sigma = p.sigma2_noise ** 0.5
    t1 = exp_minus_At.dot(S0).dot(exp_minus_At)

    s0 = p.sigma2_noise * p.tau_y / 2. * (1 - np.exp(-2 / p.tau_y * t))

    s1_1 = (p.sigma2_noise * p.tau_x * p.tau_y ** 2) / (2 * p.C * (p.tau_x ** 2 - p.tau_y ** 2))
    s1_2 = 2 * p.tau_x * (1 - np.exp(-t * (1 / p.tau_x + 1 / p.tau_y)))
    s1_3 = (p.tau_x + p.tau_y) * (1 - np.exp(-2. * t / p.tau_y))
    s1 = s1_1 * (s1_2 - s1_3)

    s2_1 = s1_1 * p.tau_x / (p.C * (p.tau_x - p.tau_y))
    s2_2 = (p.tau_x ** 2 + p.tau_x * p.tau_y) * (1 - np.exp(-2 / p.tau_x * t))
    s2_3 = (p.tau_y ** 2 + p.tau_x * p.tau_y) * (1 - np.exp(-2 / p.tau_y * t))
    s2_4 = 4 * p.tau_x * p.tau_y * (1 - np.exp(- t * (1 / p.tau_x + 1 / p.tau_y)))
    s2 = s2_1 * (s2_2 + s2_3 - s2_4)
    t2 = np.array([[s0, s1],
                   [s1, s2]])

    return t1 + t2


m0 = np.zeros(2, dtype=np.float64)
m0[0] = 3.
m0[1] = 0.3
S0 = np.zeros((2,2), dtype=np.float64)
S0[0,0] = 1.3 
S0[0,1] = 0.2
S0[1,0] = 0.2
S0[1,1] = 0.3

ts = np.arange(0, t, p.dt)
mu_soln = []
S_soln = []
for t_ in ts:
    mu = compute_expectation(m0, p, t_) 
    mu_soln.append(mu.squeeze())

    S = compute_cov(S0, p, t_)
    S_soln.append(S)

mu_soln = np.array(mu_soln)
S_soln = np.array(S_soln)



z0 = multivariate_normal.rvs(mean=m0, cov=S0, size=p.num_procs)
psim = ParticleSimulator(z0, 0., p)
psim.simulate(t)

gs = GridSpec(2, 6)
phi = np.arctan(1080 / 1920)
sz = (14 * np.cos(phi), 14 * np.sin(phi)),  
fig = plt.figure()
fig.set_size_inches(*sz)
ax1 = fig.add_subplot(gs[0,:3])
ax1.plot(psim.z[...,0].mean(1), label="simulation")
ax1.plot(mu_soln[:,0], '--', label="analytical_soln")
ax1.legend()

ax2 = fig.add_subplot(gs[0,3:])
ax2.plot(psim.z[...,1].mean(1))
ax2.plot(mu_soln[:,1], '--')

s = np.mean((psim.z[...,0] - psim.z[...,0].mean(1, keepdims=True)) * (psim.z[...,1] - psim.z[...,1].mean(1, keepdims=True)), axis=1)
ax3 = fig.add_subplot(gs[1,2:4])
ax3.plot(s)
ax3.plot(S_soln[:,0,1], '--')


ax4 = fig.add_subplot(gs[1,:2])
ax4.plot(psim.z[...,0].var(1))
ax4.plot(S_soln[:,0,0], '--')

ax4 = fig.add_subplot(gs[1,4:])
ax4.plot(psim.z[...,1].var(1))
ax4.plot(S_soln[:,1,1], '--')

gs.update(wspace=1.)
plt.show()



