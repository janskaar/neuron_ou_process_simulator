import numpy as np
from scipy.special import erf, erfc
from scipy.linalg import expm
from scipy.stats import norm


def xy_to_xv(mu, s, p):
    new_s = s.copy()
    new_mu = mu.copy()

    new_mu[:,0] = -mu[:,1] / p.tau_x + mu[:,0] / p.C
    new_s[:,0] = s[:,2] / p.tau_x ** 2 + s[:,0] / p.C ** 2\
            - 2 / (p.tau_x * p.C) * s[:,1] 

    new_s[:,1] = -s[:,2] / p.tau_x + s[:,1] / p.C
    return new_mu, new_s

def f1_schwalger(b, b_dot, mu_xy, s_xy, p):
    det = s_xy[:,0] * s_xy[:,2] - s_xy[:,1] ** 2

    B_1 = (s_xy[:,2] / p.tau_x ** 2 - 2 * s_xy[:,1] / p.tau_x + s_xy[:,0]) * b  ** 2
    B_2 = 2 * (s_xy[:,2] / p.tau_x - s_xy[:,1]) * b * b_dot
    B_3 = s_xy[:,2] * b_dot ** 2

    B = (B_1 + B_2 + B_3) / (2 * det)

    H_arg = ((s_xy[:,2] / p.tau_x - s_xy[:,1]) * b + s_xy[:,2] * b_dot) / np.sqrt(2 * det * s_xy[:,2])

    H = 1 - np.sqrt(np.pi) * H_arg * np.exp(H_arg ** 2) * erfc(H_arg)
    f = np.sqrt(det) / (2 * np.pi * s_xy[:,2]) * H * np.exp(-B)
    np.nan_to_num(f, copy=False)
    return f

def integral_f1_xdot(b, b_dot, mu, s):
    if len(mu.shape) == 1:
        mu = mu[None,:]

    if len(s.shape) == 1:
        s = s[None,:]

    mu_v = mu[:,0]
    mu_x = mu[:,1]
    
    s_vv = s[:,0]
    s_xv = s[:,1]
    s_xx = s[:,2]

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

    if len(mu_xv.shape) == 1:
        mu_xv = mu_xv[None,:]

    if len(s_xv.shape) == 1:
        s_xv = s_xv[None,:]

    f1 = integral_f1_xdot(b, b_dot, mu_xv, s_xv)
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

    f1 = integral_f1_xdot(b, b_dot, mu_xv, s_xv)
    prob_b = pdf_b(b, mu_xv[:,1], s_xv[:,2])
    return f1 * prob_b



def compute_p_y_upcrossing(b, b_dot, mu_xv, s_xv, n1, num=51):
    #
    # ASSUMES ONLY A SINGLE TIME STEP, NOT ENTIRE TIME SERIES
    #
    mu_v_x = mu_xv[0] + s_xv[1] / s_xv[2] * (b - mu_xv[1])
    s_v_x = s_xv[0] - s_xv[1] ** 2 / s_xv[2]
    p_b = norm.pdf(b, loc=mu_xv[0], scale=s_xv[2] ** 0.5)

    print("XXXXXXXXXXXXXXXXXXXX")
    print(f"MU = {mu_v_x}")
    print(f"S = {s_v_x}")
    print("XXXXXXXXXXXXXXXXXXXX")

    vs = np.linspace(b_dot, mu_v_x + 5 * s_v_x ** 0.5, num)
    E_v = np.trapz(norm.pdf(vs, loc=mu_v_x, scale = s_v_x ** 0.5) * vs ** 2, x=vs) * p_b / n1
    E_v2 = np.trapz(norm.pdf(vs, loc=mu_v_x, scale = s_v_x ** 0.5) * vs ** 3, x=vs) * p_b / n1

    return E_v, E_v2

