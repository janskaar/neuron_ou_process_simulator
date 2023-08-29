import numpy as np
from scipy.special import erf, erfc
from scipy.linalg import expm
from scipy.stats import norm

def ignore_numpy_warnings(func):

    def wrapper(*args, **kwargs):
        with np.errstate(invalid="ignore", divide="ignore"):
            return func(*args, **kwargs)

    return wrapper


def xv_to_xy(mu, s, p):
    if len(mu.shape) == 1:
        mu = mu[None,:]

    if len(s.shape) == 1:
        s = s[None,:]

    new_s = s.copy()
    new_mu = mu.copy()

    new_mu[:,0] = mu[:,0] * p.C + mu[:,1] / p.R

    new_s[:,0] = s[:,0] * p.C ** 2 + 2 * p.C ** 2 / p.tau_x * s[:,1] + s[:,2] / p.R ** 2
    new_s[:,1] = s[:,1] * p.C + s[:,2] / p.R
    return new_mu.squeeze(), new_s.squeeze()


def xy_to_xv(mu, s, p):
    if len(mu.shape) == 1:
        mu = mu[None,:]

    if len(s.shape) == 1:
        s = s[None,:]

    new_s = s.copy()
    new_mu = mu.copy()

    new_mu[:,0] = -mu[:,1] / p.tau_x + mu[:,0] / p.C
    new_s[:,0] = s[:,2] / p.tau_x ** 2 + s[:,0] / p.C ** 2\
            - 2 / (p.tau_x * p.C) * s[:,1] 

    new_s[:,1] = -s[:,2] / p.tau_x + s[:,1] / p.C

    return new_mu.squeeze(), new_s.squeeze()

@ignore_numpy_warnings
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


def gaussian_pdf(x, mu, s):
    """
    Computes the pdf of a univariate Gaussian distribution.

    """

    if len(np.shape(mu)) != len(np.shape(s)):
        raise ValueError

    # Ensure mu and s is shape (t,)
    if len(np.shape(mu)) == 0:
        mu = np.array([mu]) 
        s = np.array([s])
    elif len(np.shape(mu)) == 1:
        pass
    else:
        raise ValueError("mu and s must be either scalars or vectors")

    # Allow x to broadcast along dim 0 if necessary
    if len(np.shape(x)) == 0:
        pass
    elif len(np.shape(x)) == 1:
        mu = mu[:,None]
        s = s[:,None]
        x = x[None,:]
    else:
        raise ValueError

    return 1. / ((2. * np.pi * s) ** 0.5) * np.exp(-0.5 / s * (x - mu) ** 2)
    


def conditional_bivariate_gaussian_pdf(y, x, mu_xy, s_xy):
    """
    Given a bivariate Gaussian distribution over X(t) and Y(t), this function 
    computes the conditional distribution p(y(t)|x(t) = X(t)).

    mu_xy is the expectation, where the first column is E[Y(t)], and the
    second element E[X(t)].

    s_xy is the covariance, where the first element is Var(Y(t)), the second
    element is Cov(X(t), Y(t)) and the third element is Var(X(t)).

    if mu_xy and s_xy has shape (2,) and (3,), it will be reshaped to
    (1, 2) and (1, 3) respectively.

    x can have shape () or (t,). In the first case, it will be reshaped to (t,).

    y can have shape (), (t,) or (N, t). In the second case, mu_xy and s_xy
    must have shape(t, 2) and (t, 3) respectively. In the third case, mu_xy,
    s_xy and x will be reshaped to (1, t, 2), (1, t, 3) and (1, t) to broadcast
    appropriately.

    """

    if len(mu_xy.shape) != len(s_xy.shape):
        raise ValueError("mu and s must have same number of dims")

    # Ensure mu_xy and s_xy is shape (t,2) and (t,3)
    if len(mu_xy.shape) == 1:
        mu_xy = mu_xy[None,:]
        s_xy = mu_xy[None,:]

    # Ensure x has shape (t,)
    if len(np.shape(x)) == 0:
        x = np.array([x])
    elif len(np.shape(x)) == 1:
        pass
    else:
        raise ValueError("x must be scalar or vector")


    # Allow x to broadcast along dim 0 if necessary
    mu_x_y = mu_xy[:,0] + s_xy[:,1] / s_xy[:,2] * (x - mu_xy[:,1])
    s_x_y = s_xy[:,0] - s_xy[:,1] ** 2 / s_xy[:,2]

    if len(np.shape(y)) == 0:
        pass
    elif len(np.shape(y)) == 1:
        mu_x_y = mu_x_y[:,None]
        s_x_y = s_x_y[:,None]
        y = y[None,:]
    else:
        raise ValueError("x must be a vector or scalar")

    pdf_x_y = 1. / ((2. * np.pi * s_x_y) ** 0.5) * np.exp(-0.5 / s_x_y * (y - mu_x_y) ** 2)
    return pdf_x_y

def conditional_bivariate_gaussian(x, mu_xy, s_xy):
    """
    Given a bivariate Gaussian distribution over X(t) and Y(t), this function 
    computes the conditional distribution p(y(t)|x(t) = X(t)).

    mu_xy is the expectation, where the first column is E[Y(t)], and the
    second element E[X(t)].

    s_xy is the covariance, where the first element is Var(Y(t)), the second
    element is Cov(X(t), Y(t)) and the third element is Var(X(t)).

    if mu_xy and s_xy has shape (2,) and (3,), it will be reshaped to
    (1, 2) and (1, 3) respectively.

    x can have shape () or (t,). In the first case, it will be reshaped to (t,).

    y can have shape (), (t,) or (N, t). In the second case, mu_xy and s_xy
    must have shape(t, 2) and (t, 3) respectively. In the third case, mu_xy,
    s_xy and x will be reshaped to (1, t, 2), (1, t, 3) and (1, t) to broadcast
    appropriately.

    """

    if len(mu_xy.shape) != len(s_xy.shape):
        raise ValueError("mu and s must have same number of dims")

    # Ensure mu_xy and s_xy is shape (t,2) and (t,3)
    if mu_xy.shape == (2,):
        mu_xy = mu_xy[None,:]
        s_xy = s_xy[None,:]

    # Ensure x has shape (t,)
    if len(np.shape(x)) == 0:
        x = np.array([x])
    elif len(np.shape(x)) == 1:
        pass
    else:
        raise ValueError("x must be scalar or vector")


    # Allow x to broadcast along dim 0 if necessary
    mu_x_y = mu_xy[:,0] + s_xy[:,1] / s_xy[:,2] * (x - mu_xy[:,1])
    s_x_y = s_xy[:,0] - s_xy[:,1] ** 2 / s_xy[:,2]

    return mu_x_y, s_x_y


@ignore_numpy_warnings
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
 

@ignore_numpy_warnings
def pdf_b(b, mu_x, s_xx):
    pdf = 1 / np.sqrt(2 * np.pi * s_xx) * np.exp(-(b - mu_x) ** 2 / (2 * s_xx))
    np.nan_to_num(pdf, copy=False)
    return pdf
 
@ignore_numpy_warnings
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

@ignore_numpy_warnings
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


@ignore_numpy_warnings
def compute_mu_var_v_upcrossing(b, b_dot, mu_xv, s_xv, n1, num=51):
    #
    # ASSUMES ONLY A SINGLE TIME STEP, NOT ENTIRE TIME SERIES
    #
    mu_v_x = mu_xv[0] + s_xv[1] / s_xv[2] * (b - mu_xv[1])
    s_v_x = s_xv[0] - s_xv[1] ** 2 / s_xv[2]
    p_b = norm.pdf(b, loc=mu_xv[0], scale=s_xv[2] ** 0.5)

    vs = np.linspace(b_dot, mu_v_x + 5 * s_v_x ** 0.5, num)
    E_v = np.trapz(norm.pdf(vs, loc=mu_v_x, scale = s_v_x ** 0.5) * vs ** 2, x=vs) * p_b / n1
    E_v2 = np.trapz(norm.pdf(vs, loc=mu_v_x, scale = s_v_x ** 0.5) * vs ** 3, x=vs) * p_b / n1

    return E_v, E_v2 - E_v ** 2

@ignore_numpy_warnings
def compute_p_v_upcrossing(v, b, b_dot, mu_xv, s_xv):
    if (v.min() < b_dot).any():
        raise ValueError("v outside support")

    mu_v_b, s_v_b = conditional_bivariate_gaussian(b, mu_xv, s_xv)

    f1 = integral_f1_xdot(b, b_dot, mu_xv, s_xv)
    p = gaussian_pdf(v, mu_v_b, s_v_b)
    return p * (v - b_dot) / f1

@ignore_numpy_warnings
def ou_soln_upcrossing_y_delta_x(v, mu_0, s_0, f_0, mu_t, s_t, b_dot):
    f1 = np.sqrt(np.pi / 2) * (s_0 * s_t) ** 0.5 / (s_0 + s_t) ** 1.5 \
            * np.exp(-0.5 * (y - (mu_0 - mu_t)) ** 2 / (s_0 + s_t))

    arg_sq = -0.5 * (mu_0 * s_t + (y + mu_t) * s_0 - b_dot * (s_0 + s_t)) ** 2 \
            (s_0 * s_t * (s_0 * s_t))

    arg = arg_sq ** 0.5
    
    return f1 * ((mu_0 * s_t + y * s_0 + mu_t * s_0 - b * (s_0 * s_t * (s_0 * s_t)) ** 0.5)\
            * erfc(arg) + np.exp(arg_sq))











