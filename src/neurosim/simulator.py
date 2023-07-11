import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from scipy.linalg import expm
from scipy.integrate import cumtrapz
from scipy.stats import norm

from scipy.special import erf, erfc
from dataclasses import dataclass


PSD_EPS = 1e-8


@dataclass
class SimulationParameters:
    dt: float = 0.1
    num_procs: int = 1000000
    
    sigma_noise: float = 1.0
    tau_x: float = 20.0
    tau_y: float = 4.0
    C: float = 250.0
    mu: float = 1.0
    threshold: float = 1.0
    E_L: float = 0.0
    I_e: float = 0.


    @property
    def sigma2_noise(self):
        return self.sigma_noise**2


    @property
    def tau_tilde(self):
        return 1 / (1 / self.tau_x + 1 / self.tau_y)


    @property
    def R(self):
        return self.tau_x / self.C


class Simulator:

    def __init__(self,
                 params=None,
                 seed=1234,
                 crossing_times=None,
                 mu0=None,
                 cov0=None,
                 u0=None):


        np.random.seed(seed)

        if params is None:
            self.p = SimulationParameters()
        else:
            self.p = params


        self.set_up_propagators()

        if mu0 is None:
            self.mu0 = np.array([0., 0.])
        else:
            self.mu0 = mu0

        if cov0 is None:
            self.cov0 = np.array([[0., 0.],
                                  [0., 0.]]) + np.eye(2) * PSD_EPS
        else:
            self.cov0 = cov0

        if u0 is None:
            self.u0 = 0.

        if crossing_times is None:
            self.crossing_times = []
            self.crossing_inds = []
        else:
            self.crossing_times = crossing_times
            self.crossing_inds = [int(t/self.p.dt) for t in crossing_times]

        self._step = 0

        self.crossing_mu_y = []
        self.crossing_mu_x = []

        self.crossing_s_yy = []
        self.crossing_s_xy = []
        self.crossing_s_xx = []

        self.crossing_b = []
        self.crossing_b_dot = []


    def set_up_propagators(self):

        # Expectation propagator
        A = np.array([[-1./self.p.tau_y, 0.],
                      [1./self.p.C, -1./self.p.tau_x]])
        self.expectation_prop = expm(A * self.p.dt)
        
        # Covariance propagator
        B = np.array([[-2./self.p.tau_y,     0.     ,   0.   ],
                      [ 1./self.p.C    ,-1/self.p.tau_tilde,   0.   ],
                      [  0.     ,   2 / self.p.C    ,-2/self.p.tau_x]])
        self.cov_prop = expm(B * self.p.dt)
        B_inv = np.linalg.solve(B, np.eye(3))
        self.cov_prop_const = B_inv.dot(np.eye(3) - self.cov_prop)\
                .dot(np.array([[self.p.sigma2_noise,0.,0.]]).T)
        
        # u propagator
        self.u_prop = np.exp(-self.p.dt / self.p.tau_x)


    def simulate(self, t):
        num_steps = int(t / 0.1)
        self.s = np.zeros((num_steps,3), dtype=np.float64) # analytical solution to second cumulants
        self.s[0] = np.array([self.cov0[0,0], self.cov0[0,1], self.cov0[1,1]])
        self.mu = np.zeros((num_steps, 2), dtype=np.float64) # analytical solution to expectations
        self.mu[0] = self.mu0

        self.u = np.zeros(num_steps, dtype=np.float64)
        self.u[0] = self.u0

        self.b = np.zeros(num_steps, dtype=np.float64)
        self.b[0] = self.p.threshold - self.u[0]

        self.b_dot = np.zeros(num_steps, dtype=np.float64)
#        self.b_dot[0] = self.u[0] / self.p.tau_x + 
        
        self.z = np.zeros((num_steps, self.p.num_procs,  2), dtype=np.float64)
        self.z[0] = sp.stats.multivariate_normal(self.mu0, self.cov0).rvs(self.p.num_procs)

        self.upcrossings = np.zeros((num_steps, self.p.num_procs), dtype=bool)
        self.downcrossings = np.zeros((num_steps, self.p.num_procs), dtype=bool)

        for j in range(1, num_steps+1):
            i = self._step # shorthand
            print(i, end="\r")

            self.propagate()        

            self.upcrossings[i] = (self.z[i,:,1] >= self.b[i]) & (self.z[i-1,:,1] < self.b[i])
            self.downcrossings[i] = (self.z[i,:,1] < self.b[i]) & (self.z[i-1,:,1] >= self.b[i])


            if i in self.crossing_inds:

                self.crossing_mu_y.append(self.mu[i,0])
                self.crossing_mu_x.append(self.mu[i,1])

                self.crossing_s_yy.append(self.s[i,0])
                self.crossing_s_xy.append(self.s[i,1])
                self.crossing_s_xx.append(self.s[i,2])

                self.crossing_b.append(self.b[i])
                self.crossing_b_dot.append(self.b_dot[i])

                # sample y and set x=b for all particles
                mu_, s_ = self.compute_p_y_crossing()
                self.z[i,:,0] = np.random.normal(loc=mu_, scale=np.sqrt(s_), size=self.p.num_procs)
                self.z[i,:,1] = self.b[i]

                # set new expectations
                self.mu[i,0] = mu_
                self.mu[i,1] = self.b[i]

                # set new covariances
                self.s[i,0] = s_
                self.s[i,1] = 0.
                self.s[i,2] = 0.

            self._step += 1


    def propagate(self):
        i = self._step
        self.z[i] = self.expectation_prop.dot(self.z[i-1].T).T
        self.z[i,:,0] += np.random.randn(self.p.num_procs) * self.p.sigma_noise * np.sqrt(self.p.dt)

        self.s[i] = (self.cov_prop.dot(self.s[i-1][:,None]) - self.cov_prop_const).squeeze()
        self.mu[i] = self.expectation_prop.dot(self.mu[i-1][::,None]).squeeze()

        self.u[i] = self.u_prop * (self.u[i-1] - self.p.E_L) + self.p.R * self.p.I_e * (1 - self.u_prop)
        self.b[i] = self.p.threshold - self.u[i]



    @property
    def s_xx(self):
        """
        Variance of x
        """
        return self.s[:,2]


    @property
    def s_xv(self):
        """
        Covariance between x and x_dot
        """
        return -self.s[:,2] / self.p.tau_x + self.s[:,1] / self.p.C


    @property
    def s_vv(self):
        """
        Variance of x_dot
        """
        return self.s[:,2] / self.p.tau_x ** 2 + self.s[:,0] / self.p.C ** 2\
                - 2 / (self.p.tau_x * self.p.C) * self.s[:,1]


    @property
    def mu_v(self):
        return -self.mu[:,1] / self.p.tau_x + self.mu[:,0] / self.p.C


    @property
    def mu_x(self):
        return self.mu[:,1]


    def convert_to_mu_v(self, mu_y, mu_x):
        return -mu_x / self.p.tau_x + mu_y / self.p.C


    def convert_to_s_vv(self, s_yy, s_xy, s_xx):
        return s_xx / self.p.tau_x ** 2 + s_yy / self.p.C ** 2\
                - 2 / (self.p.tau_x * self.p.C) * s_xy


    def convert_to_s_xv(self, s_xy, s_xx):
        return -s_xx / self.p.tau_x + s_xy / self.p.C


#     def compute_n(self):
#         self.n = self.integral_f1_xdot() * self.pdf_b()


    def compute_log_n(self):
        log_n = 0.
        for i, _ in enumerate(self.crossing_mu_y):
            mu_y = self.crossing_mu_y[i]
            mu_x = self.crossing_mu_x[i]
            mu_v = self.convert_to_mu_v(mu_y, mu_x)

            s_yy = self.crossing_s_yy[i]
            s_xy = self.crossing_s_xy[i]
            s_xx = self.crossing_s_xx[i]
            s_vv = self.convert_to_s_vv(s_yy, s_xy, s_xx)
            s_xv = self.convert_to_s_xv(s_xy, s_xx)

            b = self.crossing_b[i]
            b_dot = self.crossing_b_dot[i]

            f1 = self.integral_f1_xdot(b, b_dot, mu_v, mu_x, s_vv, s_xv, s_xx)
            logprob_b = self.logpdf_b(b, mu_x, s_xx)
            log_n += np.log(f1) + logprob_b

        return log_n

    def compute_p_y_crossing(self):
        """
        Compute the conditional distribution p(y|x=b)
        Returns mean and variance of distribution over y.
        """
        i = self._step 
        mu_y = self.mu[i,0] + self.s[i,1] / self.s[i,2] * (self.b[i] - self.mu[i,1])  # conditional mean
        s_y = self.s[i,0] - self.s[i,1]**2 / self.s[i,2] # conditional variance
        return mu_y, s_y

#     def integral_f1_xdot(self):
#         sigma2 = self.s_vv - self.s_xv**2 / self.s_xx
#         sigma = np.sqrt(sigma2)
#         mu = self.mu_v + self.s_xv / self.s_xx * (self.b - self.mu_x)
#         t1 = sigma / np.sqrt(2*np.pi) * np.exp(-0.5*(self.b_dot-mu)**2 / sigma2)
#         t2 = 0.5 * (self.b_dot - mu) * (1 - erf((self.b_dot-mu) / (np.sqrt(2) * sigma)))
#         f1 = t1 - t2
#         np.nan_to_num(f1, copy=False)
#         return f1
    
    def integral_f1_xdot(self, b, b_dot, mu_v, mu_x, s_vv, s_xv, s_xx):
        sigma2 = s_vv - s_xv**2 / s_xx
        sigma = np.sqrt(sigma2)
        mu = mu_v + s_xv / s_xx * (b - mu_x)
        t1 = sigma / np.sqrt(2*np.pi) * np.exp(-0.5*(b_dot-mu)**2 / sigma2)
        t2 = 0.5 * (b_dot - mu) * (1 - erf((b_dot-mu) / (np.sqrt(2) * sigma)))
        f1 = t1 - t2
        np.nan_to_num(f1, copy=False)
        return f1
 
    def pdf_b(self, b, mu_x, s_xx):
        pdf = 1 / np.sqrt(2 * np.pi * s_xx) * np.exp(-(b - mu_x) ** 2 / (2 * s_xx))
        np.nan_to_num(pdf, copy=False)
        return pdf
    
    def logpdf_b(self, b, mu_x, s_xx):
        return - 0.5 * np.log(2 * np.pi * s_xx) - 0.5 * (b - mu_x) ** 2 / s_xx

    def compute_p_yx(self, mu_y_0, mu_x_0, s_yy_0, s_xy_0, s_xx_0): 
        """Compute the joint distribution p(y,x) at time t given
        initial conditions x0, assumed to be a delta distribution,
        and y0, assumed to be Gaussian with mean mu_y0 and variance s_y0
    
        Returns mean and covariance matrix.
        """
    
        print(t)
        # expectation propagator
        mu_0 = np.array([[mu_y_0, mu_x_0]]).T
        expAt = expm(A * t)
        mu = expAt.dot(mu_0) 
    
    
        # 2nd moments propagator
        s_0 = np.array([[s_yy_0, s_xy_0, s_xx_0]]).T
        expBt = expm(B * t)
        B_inv = np.linalg.solve(B, np.eye(3))
        b_const_term = B_inv.dot(np.eye(3) - expBt).dot(np.array([[sigma2_noise,0.,0.]]).T)
        s = expBt.dot(s_0) - b_const_term
        return mu.squeeze(), s.squeeze()







