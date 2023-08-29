import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
from functools import partial
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from scipy.linalg import expm
from scipy.integrate import cumtrapz
from scipy.stats import norm
from abc import ABC, abstractmethod




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


class SimulatorBase(ABC):
    def __init__(self, params):
        self.p = params
        self.set_up_propagators()

    @abstractmethod
    def set_up_propagators(self):
        pass
        eprop, covprop, covconst, uprop = self.compute_propagators(self.p.dt)
        self.expectation_prop = eprop
        self.cov_prop = covprop
        self.cov_prop_const = covconst
        self.u_prop = uprop

    def compute_propagators(self, t):
        pass
        # Expectation propagator
        A = np.array([[-1./self.p.tau_y, 0.],
                      [1./self.p.C, -1./self.p.tau_x]])
        expectation_prop = expm(A * t)
        
        # Covariance propagator
        B = np.array([[-2./self.p.tau_y,     0.     ,   0.   ],
                      [ 1./self.p.C    ,-1/self.p.tau_tilde,   0.   ],
                      [  0.     ,   2 / self.p.C    ,-2/self.p.tau_x]])
        cov_prop = expm(B * t)
        B_inv = np.linalg.solve(B, np.eye(3))
        cov_prop_const = B_inv.dot(np.eye(3) - cov_prop)\
                .dot(np.array([[self.p.sigma2_noise,0.,0.]]).T)
        
        # u propagator
        u_prop = np.exp(-t / self.p.tau_x)
        return expectation_prop, cov_prop, cov_prop_const, u_prop


class ParticleSimulator(SimulatorBase):

    def __init__(self,
                 z_0,
                 u_0,
                 params,
                 fix_x_threshold=False):

        super().__init__(params)

        self.z_0 = z_0
        self.u_0 = u_0
        self._step = 0
        self.fix_x_threshold = fix_x_threshold

    def set_up_propagators(self):
        eprop, covprop, covconst = self.compute_propagators()
        self.expectation_prop = eprop
        self.cov_prop = covprop
        self.cov_prop_const = covconst


    def compute_propagators(self):
        A = np.array([[-1./self.p.tau_y, 0.],
                      [1./self.p.C, -1./self.p.tau_x]])
        expectation_prop = expm(A * self.p.dt)
        
        # Covariance propagator
        B = np.array([[-2./self.p.tau_y,     0.     ,   0.   ],
                      [ 1./self.p.C    ,-1/self.p.tau_tilde,   0.   ],
                      [  0.     ,   2 / self.p.C    ,-2/self.p.tau_x]])
        cov_prop = expm(B * self.p.dt)
        B_inv = np.linalg.solve(B, np.eye(3))
        cov_prop_const = B_inv.dot(np.eye(3) - cov_prop)\
                .dot(np.array([[self.p.sigma2_noise,0.,0.]]).T)
        return expectation_prop, cov_prop, cov_prop_const 

    def simulate(self, t):
        num_steps = int(t / self.p.dt)
        self.num_steps = num_steps
        
        self.usim = MembranePotentialSimulator(self.u_0, self.p)
        self.usim.simulate(t)

        self.u = self.usim.u
        self.b = self.usim.b

        self.z = np.zeros((num_steps+1, self.p.num_procs,  2), dtype=np.float64)
        self.z[0] = self.z_0

        self.upcrossings = np.zeros((num_steps+1, self.p.num_procs), dtype=bool)
        self.downcrossings = np.zeros((num_steps+1, self.p.num_procs), dtype=bool)
        self._step += 1

        for _ in range(num_steps):
            i = self._step

            self.propagate()        

            self.upcrossings[i] = (self.z[i,:,1] >= self.b[i]) & (self.z[i-1,:,1] < self.b[i])
            self.downcrossings[i] = (self.z[i,:,1] < self.b[i]) & (self.z[i-1,:,1] >= self.b[i])
            

            if self.fix_x_threshold:
                self.z[...,1][i,self.upcrossings[i]] = self.b[i]

            self._step += 1

    def compute_N1_N2(self):
        """
        N1(t) denotes the number of particles with an upcrossing at time t
        N2(t, t') denotes the number of particles with an upcrossing at time t AND time t'
        """
        self.N1 = self.upcrossings.sum(1)
        self.N2 = np.zeros((self.num_steps+1, self.num_steps+1), dtype=np.float64)
        for i in range(self.num_steps):
            for j in range(i+1, self.num_steps, 1):
                self.N2[i,j] = float((self.upcrossings[i] & self.upcrossings[j]).sum())

    def propagate(self):
        i = self._step
        self.z[i] = self.expectation_prop.dot(self.z[i-1].T).T
        self.z[i,:,0] += np.random.randn(self.p.num_procs) * self.p.sigma_noise * np.sqrt(self.p.dt)

#        self.u[i] = self.u_prop * (self.u[i-1] - self.p.E_L) + self.p.R * self.p.I_e * (1 - self.u_prop)

#        self.b[i] = self.p.threshold - self.u[i]

class MembranePotentialSimulator(SimulatorBase):
    def __init__(self, u_0, params):
        super().__init__(params)
        self._step = 0 
        self.u_0 = u_0

    def simulate(self, t):

        num_steps = int(np.rint(t / self.p.dt))

        self.u = np.zeros(num_steps + 1, dtype=np.float64)
        self.b = np.zeros(num_steps + 1, dtype=np.float64)
        self.u[0] = self.u_0
        self.b[0] = self.p.threshold - self.u_0
        self._step += 1

        for _ in range(num_steps):
            self.propagate()
            self._step += 1

    def propagate(self):
        i = self._step
        self.u[i] = self.u_prop * (self.u[i-1] - self.p.E_L) + self.p.R * self.p.I_e * (1 - self.u_prop)
        self.b[i] = self.p.threshold - self.u[i]

    @property
    def b_dot(self):
        return self.u / self.p.tau_x - self.p.I_e / self.p.C

    def set_up_propagators(self):
        self.u_prop = self.compute_propagators(self.p.dt)

    def compute_propagators(self, t):
        # u propagator
        u_prop = np.exp(-t / self.p.tau_x)
        return u_prop

class MomentsSimulator(SimulatorBase):

    def __init__(self,
                 mu_0,
                 s_0,
                 params):

        super().__init__(params)

        self.mu_0 = mu_0
        self.s_0 = s_0

        self._step = 0

    def set_up_propagators(self):
        eprop, covprop, covconst = self.compute_propagators(self.p.dt)
        self.expectation_prop = eprop
        self.cov_prop = covprop
        self.cov_prop_const = covconst

    def compute_propagators(self, t):
        # Expectation propagator
        A = np.array([[-1./self.p.tau_y, 0.],
                      [1./self.p.C, -1./self.p.tau_x]])
        expectation_prop = expm(A * t)
        
        # Covariance propagator
        B = np.array([[-2./self.p.tau_y,     0.     ,   0.   ],
                      [ 1./self.p.C    ,-1/self.p.tau_tilde,   0.   ],
                      [  0.     ,   2 / self.p.C    ,-2/self.p.tau_x]])

        cov_prop = expm(B * t)
        B_inv = np.linalg.solve(B, np.eye(3))
        cov_prop_const = B_inv.dot(np.eye(3) - cov_prop)\
                .dot(np.array([[self.p.sigma2_noise,0.,0.]]).T)
        
        # u propagator
        return expectation_prop, cov_prop, cov_prop_const


    def simulate(self, t):

        num_steps = int(np.rint(t / self.p.dt))
        self.s = np.zeros((num_steps + 1, 3), dtype=np.float64) # analytical solution to second cumulants
        self.s[0] = self.s_0

        self.mu = np.zeros((num_steps + 1, 2), dtype=np.float64) # analytical solution to expectations
        self.mu[0] = self.mu_0
        self._step += 1

        for j in range(1, num_steps+1):
            i = self._step # shorthand

            self.propagate()        
            self._step += 1


    def propagate(self):
        i = self._step
        self.s[i] = (self.cov_prop.dot(self.s[i-1][:,None]) - self.cov_prop_const).squeeze()
        self.mu[i] = self.expectation_prop.dot(self.mu[i-1][::,None]).squeeze()


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



class Simulator():

    def __init__(self,
                 params=None,
                 crossing_times=None,
                 mu0=None,
                 cov0=None,
                 u0=None,
                 seed=1234):

        np.random.seed(seed)

        self.p = params
        if crossing_times is None:
            self.crossing_times = []
            self.crossing_inds = []
        else:
            self.crossing_times = crossing_times
            self.crossing_inds = [int(t/self.p.dt) for t in crossing_times]

        self.crossing_mu_y = []
        self.crossing_mu_x = []

        self.crossing_s_yy = []
        self.crossing_s_xy = []
        self.crossing_s_xx = []

        self.crossing_b = []
        self.crossing_b_dot = []

        self.n1s = []


    def compute_propagators(self, t):
        # Expectation propagator
        A = np.array([[-1./self.p.tau_y, 0.],
                      [1./self.p.C, -1./self.p.tau_x]])
        expectation_prop = expm(A * t)
        
        # Covariance propagator
        B = np.array([[-2./self.p.tau_y,     0.     ,   0.   ],
                      [ 1./self.p.C    ,-1/self.p.tau_tilde,   0.   ],
                      [  0.     ,   2 / self.p.C    ,-2/self.p.tau_x]])
        cov_prop = expm(B * t)
        B_inv = np.linalg.solve(B, np.eye(3))
        cov_prop_const = B_inv.dot(np.eye(3) - cov_prop)\
                .dot(np.array([[self.p.sigma2_noise,0.,0.]]).T)
        
        # u propagator
        u_prop = np.exp(-t / self.p.tau_x)
        return expectation_prop, cov_prop, cov_prop_const, u_prop





    def compute_n1(self, mu_0=None, s_0=None, u_0=None, return_dists=False):
        if not mu_0:
            mu_0 = np.array([0., 0.])

        if not s_0:
            s_0 = np.array([0., 0., 0.])

        if not u_0:
            u_0 = 0.

       
        n1s = []
        mus = []
        ss = []
        for t in self.crossing_times:
            mu, s, n1 = self.compute_prob_upcrossing(u_0, mu_0, s_0, t, return_dist=True)
            n1s.append(n1)
            mus.append(mu)
            ss.append(s)

        if return_dists:
            return mus, ss, n1s
        else:
            return n1s

    def simulate_n1(self, mu_0=None, s_0=None, u_0=None, return_sims=False):
        if not mu_0:
            mu_0 = np.array([0., 0.])

        if not s_0:
            s_0 = np.array([0., 0., 0.])

        if not u_0:
            u_0 = 0.

        z_0 = self.sample_z(mu_0, s_0)


        sim = ParticleSimulator(z_0, u_0, self.p) 
        sim.simulate(self.crossing_times[-1])
        n1s = []
        for i, _ in enumerate(self.crossing_times):
            n1s.append(sim.upcrossings[self.crossing_inds[i]].sum())

        if return_sims:
            return sim, n1s
        else:
            return n1s

    def simulate_n2(self, mu_0=None, s_0=None, u_0=None):
        if not mu_0:
            mu_0 = np.array([0., 0.])

        if not s_0:
            s_0 = np.array([0., 0., 0.])

        if not u_0:
            u_0 = 0.

        z_0 = self.sample_z(mu_0, s_0)
        num_t = len(self.crossing_times) 
        
        n2s = []
        for i in range(0, num_t-1, 1):
            sim_i = ParticleSimulator(z_0, u_0, self.p) 
            sim_i.simulate(self.crossing_times[i])
            upcrossings_i = sim_i.upcrossings[-1]
            mu_y_x, s_y_x = self.compute_p_yx(mu_0, s_0, self.crossing_times[i])
            mu_y_b, s_y_b = self.compute_p_y_crossing(sim_i.b[-1], mu_y_x, s_y_x)
            for j in range(i+1, num_t, 1):
                mu_0 = np.array([mu_y_b, 0.])
                s_0 = np.array([s_y_b, 0., 0.])
                z_0 = self.sample_z(mu_0, s_0)
                sim_j = ParticleSimulator(z_0, 0., self.p)
                sim_j.simulate(self.crossing_times[j] - self.crossing_times[i])
                n2s.append((sim_j.upcrossings[-1] & upcrossings_i).sum())
        return n2s

    def compute_n2(self, mu_0=None, s_0=None, u_0=None):
        if not mu_0:
            mu_0 = np.array([0., 0.])

        if not s_0:
            s_0 = np.array([0., 0., 0.])

        if not u_0:
            u_0 = 0.

        num_t = len(self.crossing_times) 
        usim = MembranePotentialSimulator(u_0, self.p)
        usim.simulate(self.crossing_times[-1])
        n2s = []
        for i in range(0, num_t-1, 1):
            t = self.crossing_times[i]
            t_ind = self.crossing_inds[i]

            p1 = self.compute_prob_upcrossing(u_0, mu_0, s_0, t)
            mu_y_x, s_y_x = self.compute_p_yx(mu_0, s_0, t)
            mu_y_b, s_y_b = self.compute_p_y_crossing(usim.b[t_ind], mu_y_x, s_y_x)

            mu_i = np.array([mu_y_b, 0.])
            s_i = np.array([s_y_b, 0., 0.])
            u_i = usim.u[t_ind]
            
            for j in range(i+1, num_t, 1):
                tdiff = self.crossing_times[j] - self.crossing_times[i]
                p2 = self.compute_prob_upcrossing(u_i, mu_i, s_i, tdiff)
                n2s.append(p1 * p2)
        return n2s


    def sample_z(self, mu, s):
        cov = np.array([[s[0], s[1]],
                        [s[1], s[2]]])

        z = sp.stats.multivariate_normal.rvs(mean=mu, cov=cov, size=self.p.num_procs)

        return z

    def simulate_mu_s(self, t):
        mu_0 = np.array([0., 0.])
        s_0 = np.array([0., 0., 0.])
        sim = MomentsSimulator(mu_0, s_0, self.p)
        sim.simulate(t)
        return sim.mu, sim.s


    def compute_prob_upcrossing(self, u_0, mu_0, s_0, t, return_dist=False):
        t_index = int(t / self.p.dt)
        mu_yx_t, s_yx_t = self.compute_p_yx(mu_0, s_0, t)
        mu_y_t = mu_yx_t[0]
        mu_x_t = mu_yx_t[1]
        mu_v_t = self.convert_to_mu_v(mu_y_t, mu_x_t)
        
        s_yy_t = s_yx_t[0]
        s_xy_t = s_yx_t[1]
        s_xx_t = s_yx_t[2]
        s_xv_t = self.convert_to_s_xv(s_xy_t, s_xx_t)
        s_vv_t = self.convert_to_s_vv(s_yy_t, s_xy_t, s_xx_t)

        usim = MembranePotentialSimulator(u_0, self.p)
        usim.simulate(t)
        b_t = usim.b[t_index]
        b_dot_t = usim.b_dot[t_index]

        f1 = self.integral_f1_xdot(b_t, b_dot_t, mu_v_t, mu_x_t, s_vv_t, s_xv_t, s_xx_t)
        prob_b = self.pdf_b(b_t, mu_x_t, s_xx_t)

        if return_dist:
            return mu_yx_t, s_yx_t, f1 * prob_b
        else:
            return f1 * prob_b


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

    def convert_to_mu_v(self, mu_y, mu_x):
        return -mu_x / self.p.tau_x + mu_y / self.p.C


    def convert_to_s_vv(self, s_yy, s_xy, s_xx):
        return s_xx / self.p.tau_x ** 2 + s_yy / self.p.C ** 2\
                - 2 / (self.p.tau_x * self.p.C) * s_xy


    def convert_to_s_xv(self, s_xy, s_xx):
        return -s_xx / self.p.tau_x + s_xy / self.p.C


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

    def compute_p_yx(self, mu_0, s_0, t): 
        """Compute the joint distribution p(y,x) at time t given
        initial conditions x0, assumed to be a delta distribution,
        and y0, assumed to be Gaussian with mean mu_y0 and variance s_y0
    
        Returns mean and covariance matrix.
        """
        mu_y_0 = mu_0[0]
        mu_x_0 = mu_0[1]

        s_yy_0 = s_0[0]
        s_xy_0 = s_0[1]
        s_xx_0 = s_0[2]

        e_prop, cov_prop, cov_const_term, u_prop = self.compute_propagators(t)
        mu_0 = np.array([[mu_y_0, mu_x_0]]).T
        mu = e_prop.dot(mu_0) 
    
        # 2nd moments propagator
        s_0 = np.array([[s_yy_0, s_xy_0, s_xx_0]]).T
        s = cov_prop.dot(s_0) - cov_const_term
        return mu.squeeze(), s.squeeze()

    def compute_p_y_crossing(self, b, mu, s):
        """
        Compute the conditional distribution p(y|x=b)
        Returns mean and variance of distribution over y.
        """
        mu_y = mu[0] + s[1] / s[2] * (b - mu[1])  # conditional mean
        s_y = s[0] - s[1]**2 / s[2] # conditional variance
        return mu_y, s_y

