import numpy as np
from scipy.special import erf, erfc
from scipy.linalg import expm


def xy_to_xv(mu, s, p):
    new_s = s.copy()
    new_mu = mu.copy()

    new_mu[:,0] = -mu[:,1] / p.tau_x + mu[:,0] / p.C
    new_s[:,0] = s[:,2] / p.tau_x ** 2 + s[:,0] / p.C ** 2\
            - 2 / (p.tau_x * p.C) * s[:,1] 

    new_s[:,1] = -s[:,2] / p.tau_x + s[:,1] / p.C
    return new_mu, new_s


def integral_f1_xdot(b, b_dot, mu_v, mu_x, s_vv, s_xv, s_xx):
    sigma2 = s_vv - s_xv**2 / s_xx
    sigma = np.sqrt(sigma2)
    mu = mu_v + s_xv / s_xx * (b - mu_x)
    t1 = sigma / np.sqrt(2*np.pi) * np.exp(-0.5*(b_dot-mu)**2 / sigma2)
    t2 = 0.5 * (b_dot - mu) * (1 - erf((b_dot-mu) / (np.sqrt(2) * sigma)))
    f1 = t1 - t2
    np.nan_to_num(f1, copy=False)
    return f1
 
def pdf_b(b, mu_x, s_xx):
    pdf = 1 / np.sqrt(2 * np.pi * s_xx) * np.exp(-(b - mu_x) ** 2 / (2 * s_xx))
    np.nan_to_num(pdf, copy=False)
    return pdf
 
def compute_n1(b, b_dot, mu_xv, s_xv):
    """
    Computes n1(t). Assumes x to be the membrane potential,
    and v to be dx/dt.
    b:      boundary value,         array size (num_steps, 1)
    b_dot:  derivative of boundary, array size (num_steps, 1)
    mu_xv:  expectation of p(v, x), array size (num_steps, 2)
            mu_xv[:,0] is expectation of p(v)
            mu_xv[:,1] is expectation of p(x)
    s_xv:   covariance of p(v, x),  array size (num_steps, 3)
            s_xv[:,0] is Var(V)
            s_xv[:,1] is Cov(V, X)
            s_xv[:,2] is Var(X)
    """

    f1 = integral_f1_xdot(b, b_dot, mu_xv[:,0], mu_xv[:,1], s_xv[:,0], s_xv[:,1], s_xv[:,2])
    prob_b = pdf_b(b, mu_xv[:,1], s_xv[:,2])
    return f1 * prob_b

def compute_n2(b, b_dot, mu_xv, s_xv):
    """
    Computes n2(t1, t2). Assumes x to be the membrane potential,
    and v to be dx/dt.
    b:      boundary value,         array size (num_steps, 1)
    b_dot:  derivative of boundary, array size (num_steps, 1)
    mu_xv:  expectation of p(v, x), array size (num_steps, num_steps, 2)
            mu_xv[:,0] is expectation of p(v)
            mu_xv[:,1] is expectation of p(x)
    s_xv:   covariance of p(v, x),  array size (num_steps, num_steps, 3)
            s_xv[:,0] is Var(V)
            s_xv[:,1] is Cov(V, X)
            s_xv[:,2] is Var(X)
    """

    f1 = integral_f1_xdot(b, b_dot, mu_xv[:,0], mu_xv[:,1], s_xv[:,0], s_xv[:,1], s_xv[:,2])
    prob_b = pdf_b(b, mu_xv[:,1], s_xv[:,2])
    return f1 * prob_b

def compute_p_y_crossing(b, mu, s):
    """
    Compute the conditional distribution p(y|x=b)
    Returns mean and variance of distribution over y.
    """
    mu_y = mu[0] + s[1] / s[2] * (b - mu[1])  # conditional mean
    s_y = s[0] - s[1]**2 / s[2] # conditional variance
    return mu_y, s_y
