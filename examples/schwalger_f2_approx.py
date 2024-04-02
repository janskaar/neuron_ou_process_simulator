import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from functools import partial
from scipy.integrate import cumtrapz
from scipy.stats import multivariate_normal, norm
from scipy.special import erfc, erf
import sys, os, time
sys.path.append("/home/janeirik/Repositories/neuron_ou_process_simulator/src")
from neurosim.simulator import SimulationParameters, MomentsSimulator, MembranePotentialSimulator, ParticleSimulator
from neurosim.n_functions import compute_n1, xv_to_xy, xy_to_xv, integral_f1_xdot
from neurosim.n_functions import compute_p_v_upcrossing, conditional_bivariate_gaussian
from neurosim.n_functions import ou_soln_xv_after_upcrossing, ou_soln_marginal_x_after_upcrossing, ou_soln_marginal_v_after_upcrossing
from pathlib import Path

path = Path(__file__).parent

phi = np.arctan(1080 / 1920)
sz = (14 * np.cos(phi), 14 * np.sin(phi))

######################
# Run simulations
######################

p = SimulationParameters(threshold=10., dt=0.01, I_e = 1., num_procs=100000, sigma_noise=0.5, tau_y = 4., C=1)
D = p.tau_y * p.sigma2_noise / p.C ** 2

t = 20.
t_vec = np.arange(0, t+p.dt, p.dt)

stim = 0.5 * np.sin(t_vec)

# moments
mu_0 = np.zeros(2, dtype=np.float64)
s_0 = np.zeros(3, dtype=np.float64)
msim = MomentsSimulator(mu_0, s_0, p)
msim.simulate(t)
mu_xy = msim.mu
s_xy = msim.s


# particles
m0 = np.zeros(2, dtype=np.float64)
sigma2_y = D / p.tau_y
S0 = np.zeros((2,2), dtype=np.float64)
S0[0,0] = p.sigma2_noise
S0[0,1] = 0.
S0[1,0] = 0.
S0[1,1] = 0.
z0 = multivariate_normal.rvs(mean=m0, cov=S0, size=p.num_procs)
psim = ParticleSimulator(z0, 0., p, stim=stim)
psim.simulate(t)


xy = psim.z.copy()
xy[...,0] /= p.C
s = np.mean((xy[...,0] - xy[...,0].mean(1, keepdims=True)) * (xy[...,1] - xy[...,1].mean(1, keepdims=True)), axis=1)

def compute_cov(p, t):
    tau_tilde = 1 / (1 / p.tau_x + 1 / p.tau_y)
    s_xy = tau_tilde * sigma2_y * (1 - np.exp(-t / tau_tilde))

    s_xx = tau_tilde * sigma2_y * p.tau_x * (1 - np.exp(-2 * t / p.tau_x)) \
        + (2 * tau_tilde * sigma2_y) / (2 / p.tau_x - 1 / tau_tilde) \
          * (np.exp(-2 * t / p.tau_x) - np.exp(-t / tau_tilde))
    return np.array([[sigma2_y, s_xy], [s_xy, s_xx]])

ts = np.arange(0, t + p.dt * 0.5, p.dt)
S_soln = []
for t_ in ts:
    S = compute_cov(p, t_)
    S_soln.append(S)

S_soln = np.array(S_soln)
sigmas = np.array([S_soln[:,0,0], S_soln[:,0,1], S_soln[:,1,1]])

xv = xy.copy()
xv[...,0] -= xv[...,1] / p.tau_x

x_dots = (xy[1:,:,1] - xy[:-1,:,1]) / p.dt

num = 101
xs = np.linspace(-60, 60, num)
ys = np.linspace(-15, 15, num)
vs = np.linspace(-15, 15, num)
XX, VV = np.meshgrid(xs, vs)
vx_grid = np.stack((VV, XX)).transpose(1,2,0).reshape((-1,2))
vx_grid_shifted = vx_grid.copy()
vx_grid_shifted[:,0] += vx_grid_shifted[:,1] / p.tau_x

XX, YY = np.meshgrid(xs, vs)
yx_grid = np.stack((YY, XX)).transpose(1,2,0).reshape((-1,2))

from scipy.stats import multivariate_normal
ind = 900
cov = np.array([[sigmas[0,ind], sigmas[1,ind]], [sigmas[1,ind], sigmas[2,ind]]])
mvn = multivariate_normal(mean=np.zeros(2), cov=cov)

pdf = mvn.pdf(vx_grid_shifted).reshape((num, num))
#plt.scatter(xv[ind,:,0], xv[ind,:,1], s=1)
plt.scatter(x_dots[ind], xv[ind,:,1], s=1)
plt.contour(VV, XX, pdf)
plt.show()

pdf = mvn.pdf(yx_grid).reshape((num, num))
plt.scatter(xy[ind,:,0], xy[ind,:,1], s=1)
plt.contour(YY, XX, pdf)
plt.show()


upcrossing_inds = (xv[ind,:,1] < psim.b[ind]) \
                & (xv[ind,:,0] > psim.b_dot[ind])\
                & (xv[ind,:,1] > psim.b[ind] - (xv[ind,:,0] - psim.b_dot[ind]) * p.dt)



gs = GridSpec(2, 6)
phi = np.arctan(1080 / 1920)
sz = (14 * np.cos(phi), 14 * np.sin(phi)),  
fig = plt.figure()
fig.set_size_inches(*sz)
s = np.mean((xy[...,0] - xy[...,0].mean(1, keepdims=True)) * (xy[...,1] - xy[...,1].mean(1, keepdims=True)), axis=1)
ax3 = fig.add_subplot(gs[1,2:4])
ax3.plot(s)
ax3.plot(S_soln[:,1,0], '--')
ax3.plot(S_soln[:,0,1], '--')

ax4 = fig.add_subplot(gs[1,:2])
ax4.plot(xy[...,0].var(1))
ax4.plot(S_soln[:,0,0], '--')

ax4 = fig.add_subplot(gs[1,4:])
ax4.plot(xy[...,1].var(1))
ax4.plot(S_soln[:,1,1], '--')

gs.update(wspace=1.)
plt.show()

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

    H_arg = ( (s[2] / p.tau_x - s[1]) * b + s[2] * b_dot ) \
           /( np.sqrt(2 * (s[2] * s[0] - s[1] ** 2) * s[2]) )

    B = schwalger_B(b, b_dot, s, p)
    return factor1 * schwalger_H(H_arg) * np.exp(-B)

def schwalger_f2_tt(b, b_dot, s, p):
    t1 = (3 * np.sqrt(3) - np.pi) / (36 * np.pi ** 2)
    t2 = sigma2_y / (p.tau_y * np.sqrt(s[2] * s[0] - s[1] ** 2)) \
            * np.exp(-schwalger_B(b, b_dot, s, p))
    return t1 * t2

def schwalger_R0(b, b_dot, s, p):
    f2 = schwalger_f2_tt(b, b_dot, s, p)
    f1 = schwalger_f1(b, b_dot, s, p)
    return f2 / f1 ** 2 - 1

def schwalger_q_approx(b, b_dot, s, p):
    t_corr = p.tau_x + p.tau_y
    f1 = schwalger_f1(b, b_dot, s, p)
    T = (len(s[0])-1) * p.dt 
    t = np.arange(0, T+0.5*p.dt, p.dt)

    integrand = f1 * np.exp(t / t_corr)
    integrand[np.isnan(integrand)] = 0.

    z = np.exp(-t / t_corr) * cumtrapz(integrand, initial=0., x=t)
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
bins = np.arange(0, t+p.dt, p.dt) - 0.5 * p.dt
fptd, _ = np.histogram(psim.fpt, bins=bins)
fptd[0] = 0
fptd = fptd / ( p.dt * p.num_procs )

sim_lambda = psim.upcrossings.sum(1) / p.num_procs / p.dt
surv = (1 - cumtrapz(fptd, x=ts))

f1[np.isnan(f1)] = 0
f2_hazard[np.isnan(f2_hazard)] = 0
f1_fptd = f1 * np.exp(-cumtrapz(f1, x=ts, initial=0))
f2_fptd = f2_hazard * np.exp(-cumtrapz(f2_hazard, x=ts, initial=0))

f1_surv = np.exp(-cumtrapz(f1, x=ts, initial=0))
f2_surv = np.exp(-cumtrapz(f2_hazard, x=ts, initial=0))


plt.plot(surv)
plt.plot(f1_surv, "--")
plt.plot(f2_surv, "--")
plt.show()


plt.plot(fptd)
plt.plot(f1_fptd, "C1")
plt.plot(f2_fptd, "C2")
plt.show()

