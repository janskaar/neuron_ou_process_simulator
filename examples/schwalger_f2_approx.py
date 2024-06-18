"""
Implement the f_1 and f_2 approximations from (Schwalger, 2021), and
compare the results to simulations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import cumtrapz
from scipy.stats import multivariate_normal, norm
from scipy.special import erfc, erf
import sys, os, time

sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import (
    SimulationParameters,
    MomentsSimulator,
    MembranePotentialSimulator,
    ParticleSimulator,
)
from neurosim.n_functions import compute_n1, xv_to_xy, xy_to_xv, integral_f1_xdot
from neurosim.n_functions import compute_p_v_upcrossing, conditional_bivariate_gaussian
from neurosim.n_functions import (
    ou_soln_xv_after_upcrossing,
    ou_soln_marginal_x_after_upcrossing,
    ou_soln_marginal_v_after_upcrossing,
)
from pathlib import Path

path = Path(__file__).parent

phi = np.arctan(1080 / 1920)
sz = (14 * np.cos(phi), 14 * np.sin(phi))

######################
# Run simulations
######################

tau_y = 4.0
sigma2_y = 0.5
sigma2_noise = sigma2_y * 2.0 / tau_y
p = SimulationParameters(
    threshold=10.0,
    dt=0.01,
    I_e=1.0,
    num_procs=10000,
    sigma_noise=np.sqrt(sigma2_noise),  # np.sqrt(sigma2_noise),
    tau_y=tau_y,
    C=1,
)

D = p.tau_y * p.sigma2_noise / p.C ** 2

t = 20.0
t_vec = np.arange(0, t + p.dt, p.dt)

stim = 0.5 * np.sin(t_vec)

# particles
m0 = np.zeros(2, dtype=np.float64)
# sigma2_y = D / p.tau_y
S0 = np.zeros((2, 2), dtype=np.float64)
S0[0, 0] = sigma2_y  # p.sigma2_noise
S0[0, 1] = 0.0
S0[1, 0] = 0.0
S0[1, 1] = 0.0
z0 = multivariate_normal.rvs(mean=m0, cov=S0, size=p.num_procs)
psim = ParticleSimulator(z0, 0.0, p, stim=stim)
psim.simulate(t)

xy = psim.z.copy()
xy[..., 0] /= p.C
s = np.mean(
    (xy[..., 0] - xy[..., 0].mean(1, keepdims=True))
    * (xy[..., 1] - xy[..., 1].mean(1, keepdims=True)),
    axis=1,
)

# solution of covariances given constant s_yy
def compute_cov(p, t):
    tau_tilde = 1 / (1 / p.tau_x + 1 / p.tau_y)
    s_xy = tau_tilde * sigma2_y * (1 - np.exp(-t / tau_tilde))

    s_xx = tau_tilde * sigma2_y * p.tau_x * (1 - np.exp(-2 * t / p.tau_x)) + (
        2 * tau_tilde * sigma2_y
    ) / (2 / p.tau_x - 1 / tau_tilde) * (
        np.exp(-2 * t / p.tau_x) - np.exp(-t / tau_tilde)
    )
    s_yy = np.full(s_xx.shape, sigma2_y)
    return np.stack((s_yy, s_xy, s_xx))
    return np.array([[s_yy, s_xy], [s_xy, s_xx]]).transpose((2, 0, 1))


ts = np.arange(0, t + p.dt * 0.5, p.dt)
sigmas = compute_cov(p, ts)


########################################
# Compute approximation from Schwalger paper
########################################


def schwalger_B(b, b_dot, s, p):
    t1 = (s[2] / p.tau_x ** 2 - 2 * s[1] / p.tau_x + s[0]) * b ** 2
    t2 = 2 * (s[2] / p.tau_x - s[1]) * b * b_dot
    t3 = s[2] * b_dot ** 2
    denom = 2 * (s[2] * s[0] - s[1] ** 2)
    return (t1 + t2 + t3) / denom


def schwalger_H(x):
    return 1 - np.sqrt(np.pi) * x * np.exp(x ** 2) * erfc(x)


def schwalger_f1(b, b_dot, s, p):
    factor1 = np.sqrt(s[2] * s[0] - s[1] ** 2) / (2 * np.pi * s[2])

    H_arg = ((s[2] / p.tau_x - s[1]) * b + s[2] * b_dot) / (
        np.sqrt(2 * (s[2] * s[0] - s[1] ** 2) * s[2])
    )

    B = schwalger_B(b, b_dot, s, p)
    return factor1 * schwalger_H(H_arg) * np.exp(-B)


def schwalger_f2_tt(b, b_dot, s, p):
    t1 = (3 * np.sqrt(3) - np.pi) / (36 * np.pi ** 2)
    t2 = (
        sigma2_y
        / (p.tau_y * np.sqrt(s[2] * s[0] - s[1] ** 2))
        * np.exp(-schwalger_B(b, b_dot, s, p))
    )
    return t1 * t2


def schwalger_R0(b, b_dot, s, p):
    f2 = schwalger_f2_tt(b, b_dot, s, p)
    f1 = schwalger_f1(b, b_dot, s, p)
    return f2 / f1 ** 2 - 1


def schwalger_q_approx(b, b_dot, s, p):
    t_corr = p.tau_x + p.tau_y
    f1 = schwalger_f1(b, b_dot, s, p)
    T = (len(s[0]) - 1) * p.dt
    t = np.arange(0, T + 0.5 * p.dt, p.dt)

    integrand = f1 * np.exp(t / t_corr)
    integrand[np.isnan(integrand)] = 0.0

    z = np.exp(-t / t_corr) * cumtrapz(integrand, initial=0.0, x=t)
    R0 = schwalger_R0(b, b_dot, s, p)
    return R0 * z


def schwalger_f2_hazard(b, b_dot, sigmas, p):
    f1 = schwalger_f1(b, b_dot, sigmas, p)
    q = schwalger_q_approx(b, b_dot, sigmas, p)
    return f1 / (1 + q)


R0 = schwalger_R0(psim.b, psim.b_dot, sigmas, p)
q_approx = schwalger_q_approx(psim.b, psim.b_dot, sigmas, p)
f2_hazard = schwalger_f2_hazard(psim.b, psim.b_dot, sigmas, p)
f1 = schwalger_f1(psim.b, psim.b_dot, sigmas, p)
psim.compute_first_passage_times()
bins = np.arange(0, t + p.dt, p.dt) - 0.5 * p.dt
fptd, _ = np.histogram(psim.fpt, bins=bins)
fptd[0] = 0
fptd = fptd / (p.dt * p.num_procs)

sim_lambda = psim.upcrossings.sum(1) / p.num_procs / p.dt
surv = 1 - cumtrapz(fptd, x=ts, initial=0)

f1[np.isnan(f1)] = 0
f2_hazard[np.isnan(f2_hazard)] = 0
f1_fptd = f1 * np.exp(-cumtrapz(f1, x=ts, initial=0))
f2_fptd = f2_hazard * np.exp(-cumtrapz(f2_hazard, x=ts, initial=0))

f1_surv = np.exp(-cumtrapz(f1, x=ts, initial=0))
f2_surv = np.exp(-cumtrapz(f2_hazard, x=ts, initial=0))


########################################
# Create plot
########################################

fig, ax = plt.subplots(2, sharex=True)

ax[0].plot(ts, fptd, c="black", label="simulation")
ax[0].plot(ts, f1_fptd, c="C1", label="f1")
ax[0].plot(ts, f2_fptd, c="C0", label="f2")
ax[0].legend()
ax[0].set_title("first passage time density")

ax[1].plot(ts, surv, c="black")
ax[1].plot(ts, f1_surv, "--", c="C1", label="f1")
ax[1].plot(ts, f2_surv, "--", c="C0", label="f2")
ax[1].set_title("survival function")
ax[1].set_xlabel("time (ms)")
ax[0].set_ylim(0, 0.2)
ax[1].set_ylim(0, 1)
fig.tight_layout()
# fig.savefig("schwalger_approximation.svg")
plt.show()
