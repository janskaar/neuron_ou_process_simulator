import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
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


def conditional_bivariate_gaussian(x, mu_xy, s_xy):
    mu_x_y = mu_xy[0] + s_xy[1] / s_xy[2] * (x - mu_xy[1])
    s_x_y = s_xy[0] - s_xy[1] ** 2 / s_xy[2]
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
def compute_p_v_upcrossing(v, b, b_dot, mu_xv, s_xv):
    if (v.min() < b_dot).any():
        raise ValueError("v outside support")

    mu_v_b, s_v_b = conditional_bivariate_gaussian(b, mu_xv, s_xv)

    f1 = integral_f1_xdot(b, b_dot, mu_xv, s_xv)
    p = gaussian_pdf(v, mu_v_b, s_v_b)
    return p * (v - b_dot) / f1



def var_v_delta_init(t, p):
    return (p.sigma2_noise * p.tau_y / 2.) * (1 - np.exp(-2 * t / p.tau_y))

def compute_v_alpha(t, p):
    return np.exp(-t / p.tau_y)



def compute_mu_terms_v_t_upcrossing_v_delta_x(t, p):
    delta_e = np.exp(-t / p.tau_x) - np.exp(-t / p.tau_y)

    v_term = np.exp(-t / p.tau_y)\
           - p.tau_y / (p.tau_x - p.tau_y) * delta_e

    x_term = -delta_e * (1 / p.tau_x + p.tau_y / (p.tau_x * (p.tau_x - p.tau_y)))

    return v_term, x_term


def compute_cov_v_t_upcrossing_v_delta_x(t, p):
    # Covariance propagator
    B = np.array([[-2./p.tau_y,     0.     ,   0.   ],
                  [ 1./p.C    ,-1/p.tau_tilde,   0.   ],
                  [  0.     ,   2 / p.C    ,-2/p.tau_x]])

    cov_prop = expm(B * t)
    B_inv = np.linalg.solve(B, np.eye(3))
    cov = -B_inv.dot(np.eye(3) - cov_prop)\
            .dot(np.array([[p.sigma2_noise,0.,0.]]).T).squeeze()
    
    var_v = cov[0] / p.C ** 2 + cov[2] / p.tau_x ** 2 - 2 * cov[1] / (p.C * p.tau_x)
    return var_v


@ignore_numpy_warnings
def ou_soln_v_upcrossing_v_delta_x(v, mu_0, s_0, f_0, b_0, b_dot_0, t, p):
    s_t = compute_cov_v_t_upcrossing_v_delta_x(t, p)
    v_term, x_term = compute_mu_terms_v_t_upcrossing_v_delta_x(t, p)

    alpha = v_term
    beta = b_0 * x_term

    arg1 = (mu_0 ** 2 * s_t\
         + (v + beta) ** 2 * s_0\
         + b_dot_0 ** 2 * (s_t + alpha ** 2 * s_0)\
         - 2 * b_dot_0 * (mu_0 * s_t + alpha * (v + beta) * s_0)\
         ) / (2 * s_t * s_0)
          
    arg2 = (mu_0 * s_t\
         + alpha * (v + beta) * s_0\
         - b_dot_0 * (s_t + alpha ** 2 * s_0)\
         ) ** 2  / (2 * s_t * s_0 * (s_t + alpha ** 2 * s_0))

    arg3 = (mu_0 * s_t\
         + alpha * (v + beta) * s_0\
         - b_dot_0 * (s_t + alpha ** 2 * s_0)\
         ) / (2 ** 0.5 * s_t * (alpha ** 2 / s_t + 1 / s_0) ** 0.5 * s_0)

    f1 = (np.pi ** 0.5) / (2 ** 0.5 * (alpha ** 2 / s_t + 1 / s_0) ** 0.5 * (s_t + alpha ** 2 * s_0) * f_0 * np.sqrt(s_t))

    e1 = np.exp(-arg1 + arg2)
    e2 = np.exp(-arg2)

    t1 = (2 / np.pi) ** 0.5 * s_t * (alpha ** 2 / s_t + 1 / s_0) ** 0.5 * s_0 * e2
    t2 = (mu_0 * s_t + alpha * (v + beta) * s_0 - b_dot_0 * (s_t + alpha ** 2 * s_0)) * (1 + erf(arg3))
    

    return f1 * e1 * (t1 + t2)

def compute_cov_x_t_upcrossing_v_delta_x(t, p):
    # Covariance propagator
    B = np.array([[-2./p.tau_y,     0.     ,   0.   ],
                  [ 1./p.C    ,-1/p.tau_tilde,   0.   ],
                  [  0.     ,   2 / p.C    ,-2/p.tau_x]])

    cov_prop = expm(B * t)
    B_inv = np.linalg.solve(B, np.eye(3))
    cov = -B_inv.dot(np.eye(3) - cov_prop)\
            .dot(np.array([[p.sigma2_noise,0.,0.]]).T).squeeze()
    
    return cov[2]

def compute_mu_terms_x_t_upcrossing_v_delta_x(t, p):
    delta_e = np.exp(-t / p.tau_x) - np.exp(-t / p.tau_y)

    v_term = (p.tau_x * p.tau_y * delta_e) / (p.tau_x - p.tau_y)
           
    x_term = (p.tau_y * delta_e) / (p.tau_x - p.tau_y) + np.exp(-t / p.tau_x)

    return v_term, x_term


@ignore_numpy_warnings
def ou_soln_x_upcrossing_v_delta_x(v, mu_0, s_0, f_0, b_0, b_dot_0, t, p):
    s_t = compute_cov_x_t_upcrossing_v_delta_x(t, p)
    v_term, x_term = compute_mu_terms_x_t_upcrossing_v_delta_x(t, p)

    alpha = v_term
    beta = -b_0 * x_term

    arg1 = (mu_0 ** 2 * s_t\
         + (v + beta) ** 2 * s_0\
         + b_dot_0 ** 2 * (s_t + alpha ** 2 * s_0)\
         - 2 * b_dot_0 * (mu_0 * s_t + alpha * (v + beta) * s_0)\
         ) / (2 * s_t * s_0)
          
    arg2 = (mu_0 * s_t\
         + alpha * (v + beta) * s_0\
         - b_dot_0 * (s_t + alpha ** 2 * s_0)\
         ) ** 2  / (2 * s_t * s_0 * (s_t + alpha ** 2 * s_0))

    arg3 = (mu_0 * s_t\
         + alpha * (v + beta) * s_0\
         - b_dot_0 * (s_t + alpha ** 2 * s_0)\
         ) / (2 ** 0.5 * s_t * (alpha ** 2 / s_t + 1 / s_0) ** 0.5 * s_0)

    f1 = (np.pi ** 0.5) / (2 ** 0.5 * (alpha ** 2 / s_t + 1 / s_0) ** 0.5 * (s_t + alpha ** 2 * s_0) * f_0 * np.sqrt(s_t))

    e1 = np.exp(-arg1 + arg2)
    e2 = np.exp(-arg2)

    t1 = (2 / np.pi) ** 0.5 * s_t * (alpha ** 2 / s_t + 1 / s_0) ** 0.5 * s_0 * e2
    t2 = (mu_0 * s_t + alpha * (v + beta) * s_0 - b_dot_0 * (s_t + alpha ** 2 * s_0)) * (1 + erf(arg3))
    
    return f1 * e1 * (t1 + t2)

def ou_soln_upcrossing_alpha_beta(t, p):
    exp_tau_x = jnp.exp(-t / p.tau_x)
    exp_tau_y = jnp.exp(-t / p.tau_y)
    delta_e = exp_tau_x - exp_tau_y
    denom1 = (1. / p.tau_y - 1. / p.tau_x)
    denom2 = denom1 * p.tau_x
    denom3 = denom2 * p.tau_x

    alpha1 = exp_tau_y - delta_e / denom2
    alpha2 = delta_e / denom1

    beta1 = -delta_e * (1 / p.tau_x + 1 / denom3)
    beta2 = exp_tau_x + delta_e / denom2

    return jnp.array([alpha1, alpha2]), jnp.array([beta1, beta2])

def ou_soln_upcrossing_S(t, p):
    # Covariance propagator
    B = jnp.array([[-2./p.tau_y,     0.        ,   0.      ],
                   [ 1./p.C    , -1/p.tau_tilde,   0.      ],
                   [  0.       ,   2 / p.C     , -2/p.tau_x]])

    cov_prop = jsp.linalg.expm(B * t)
    B_inv = jnp.linalg.solve(B, np.eye(3))
    cov = -B_inv.dot(jnp.eye(3) - cov_prop)\
            .dot(jnp.array([[p.sigma2_noise,0.,0.]]).T).squeeze()
    
    var_v = cov[0] / p.C ** 2 + cov[2] / p.tau_x ** 2 - 2 * cov[1] / (p.C * p.tau_x)
    cov_xv = cov[1] / p.C - cov[2] / p.tau_x
    S = jnp.array([[var_v, cov_xv],
                   [cov_xv, cov[2]]])

    return S


def ou_soln_xv_integrand(z, mu_0, s_0, v_0, b_0, b_dot_0, t, p):
    alpha, beta = ou_soln_upcrossing_alpha_beta(t, p)
    beta *= b_0
    mu_v_x, s_v_x = conditional_bivariate_gaussian(b_0, mu_0, s_0) 
    S = ou_soln_upcrossing_S(t, p)
    S_inv = jnp.linalg.solve(S, jnp.eye(2))

    m = (alpha * v_0 + beta)
    d = z - m  # distance
    quad2 = d.T.dot(S_inv).dot(d)

    det_S = S[0,0] * S[1,1] - S[1,0] ** 2
#     log_f1 = -0.5 * jnp.log(2 * np.pi * s_0[2])\
#              -0.5 * (b_0 - mu_0[1]) ** 2 / s_0[2]\
#              -0.5 * jnp.log(2 * np.pi * s_v_x)\
#              -0.5 * (v_0 - mu_v_x) ** 2 / s_v_x\
#              -jnp.log(2 * np.pi) - 0.5 * jnp.log(det_S)\
#              - 0.5 * quad2

    log_f1 = - 0.5 * quad2
    f = jnp.exp(log_f1) * (v_0 - b_dot_0)
    return f, quad2, S
    

# def ou_soln_xv_upcrossing_v_delta_x(z, mu_0, s_0, b_0, b_dot_0, t, p):
#     # Implementation differs from the equation in note by that we pull out
#     # exp(q^2) factor of the [1 - q exp(q^2)Erf(q)] factor
#     
#     alpha, beta = ou_soln_upcrossing_alpha_beta(t, p)
#     beta *= b_0
#     mu_v_x, s_v_x = conditional_bivariate_gaussian(b_0, mu_0, s_0) 
#     S = ou_soln_upcrossing_S(t, p)
#     S_inv = jnp.linalg.solve(S, jnp.eye(2))
#     quad1 = alpha.T.dot(S_inv).dot(alpha.T) # quadratic form of alphas
# 
#     m = (alpha * b_dot_0 + beta)
#     d = z - m  # distance
#     quad2 = d.T.dot(S_inv).dot(d)
# 
#     q = -alpha.T.dot(S_inv).dot(z - (alpha * b_dot_0 + beta)) + (b_dot_0 - mu_v_x) / s_v_x
#     q /= 2 ** 1.5 * jnp.sqrt(quad1 + 1. / s_v_x)
# 
#     det_S = S[0,0] * S[1,1] - S[1,0] ** 2
#     log_f1 = -0.5 * jnp.log(2 * np.pi * s_0[2])\
#              -0.5 * (b_0 - mu_0[1]) ** 2 / s_0[2]\
#              -0.5 * jnp.log(2 * np.pi * s_v_x)\
#              -jnp.log(2 * np.pi) - 0.5 * jnp.log(det_S)
# 
#     log_f2 = -jnp.log(quad1)
#     log_f3 = -0.5 * quad2
#     log_f4 =  q ** 2
# 
#     t1 = jnp.exp(-q ** 2)
#     t2 = np.pi ** 0.5 * q * jsp.special.erfc(q)
# 
#     f = jnp.exp(log_f1 + log_f2 + log_f3 + log_f4) * (t1 - t2)
# 
#     return f, q, quad1, quad2, log_f3

# def ou_soln_xv_upcrossing_v_delta_x(z, mu_0, s_0, b_0, b_dot_0, t, p):
#     # equation implemented directly from mathematica form
#        
#     alpha, beta = ou_soln_upcrossing_alpha_beta(t, p)
#     beta *= b_0
#     mu_v_x, s_v_x = conditional_bivariate_gaussian(b_0, mu_0, s_0) 
#     S = ou_soln_upcrossing_S(t, p)
#     S_inv = jnp.linalg.solve(S, jnp.eye(2))
# 
#     c0 = (z - beta).T.dot(S_inv).dot(z - beta) + mu_v_x ** 2 / s_v_x
#     c1 = 2 * alpha.T.dot(S_inv).dot(-z + beta) - 2 * mu_v_x / s_v_x
#     c2 = alpha.T.dot(S_inv).dot(alpha) + 1. / s_v_x
# 
#     f1 = jnp.exp(-0.5 * (c0 + b_dot_0 * (c1 + b_dot_0 * c2))) / (2 * c2) ** 1.5
# 
#     arg = (c1 + 2 * b_dot_0 * c2) / (2 ** 1.5 * c2 ** 0.5)
#     t1 = 2 ** 1.5 * c2 ** 0.5 
#     t2 = (c1 + 2 * b_dot_0 * c2) * jnp.exp(arg ** 2) * np.pi ** 0.5 * jsp.special.erfc(arg)
#     
#     return f1, t1, t2, arg, f1 * (t1 - t2)



def ou_soln_xv_after_upcrossing(z, mu_0, s_0, b_0, b_dot_0, t, p):
    # equation implemented as in note
       

    
    alpha, beta = ou_soln_upcrossing_alpha_beta(t, p)
    beta *= b_0
    mu_v_x, s_v_x = conditional_bivariate_gaussian(b_0, mu_0, s_0) 
    S = ou_soln_upcrossing_S(t, p)
    S_inv = jnp.linalg.solve(S, jnp.eye(2))


#     prefactor1 = 1 / np.sqrt(2 * np.pi * s_0[2]) * np.exp(-0.5 * (b_0 - mu_0[1]) ** 2 / s_0[2])
#     prefactor2 = 1 / np.sqrt(2 * np.pi * s_v_x)
#     prefactor3 = 1 / np.sqrt(2 * np.pi * np.linalg.det(S) ** 0.5)
#     prefactor4 = 1. / integral_f1_xdot(b_0, b_dot_0, mu_0, s_0)
#     prefactor =  prefactor2 * prefactor3 * prefactor4


    # normalizing factors
    f1 = 1. / integral_f1_xdot(b_0, b_dot_0, mu_0, s_0)  # prefactor from distribution p(v) at time 0
    n1 = 1. / (2 * np.pi * np.sqrt(np.linalg.det(S)))  # prefactor from normal dist. p(v, x) at time t
    n2 = 1. / np.sqrt(2 * np.pi * s_v_x)     # prefactor from normal dist p(v|x=b) at time 0
 
    prefactor = f1 * n1 * n2

    c0 = (z - beta).T.dot(S_inv).dot(z - beta) + mu_v_x ** 2 / s_v_x
    c1 = 2 * alpha.T.dot(S_inv).dot(-z + beta) - 2 * mu_v_x / s_v_x
    c2 = alpha.T.dot(S_inv).dot(alpha) + 1. / s_v_x

    arg_f1 = -0.5 * (c0 + b_dot_0 * (c1 + b_dot_0 * c2))
    f1 = jnp.exp(arg_f1) / c2

    arg = (c1 + 2 * b_dot_0 * c2) / (2 ** 1.5 * c2 ** 0.5)

    t2 = arg * jnp.exp(arg ** 2) * np.pi ** 0.5 * jsp.special.erfc(arg)
    
#    return arg_f1, f1 * c2, t2, arg, f1 * (1 - t2) * c2
    return prefactor * f1 * (1 - t2)



@ignore_numpy_warnings
def ou_soln_marginal_x_after_upcrossing(x, mu_0, s_0, b_0, b_dot_0, t, p):
    alpha, beta = ou_soln_upcrossing_alpha_beta(t, p)
    alpha = alpha[1]
    beta = beta[1]
    S = ou_soln_upcrossing_S(t, p)
    sigma2_t = S[1,1]
    beta *= b_0

    mu_v_x, s_v_x = conditional_bivariate_gaussian(b_0, mu_0, s_0) 

    q = (mu_v_x * sigma2_t + alpha * (x - beta) * s_v_x - b_dot_0 * (sigma2_t + alpha ** 2 * s_v_x))\
             / np.sqrt(2 * sigma2_t * s_v_x * (alpha ** 2 * s_v_x + sigma2_t))

    exparg = mu_v_x ** 2 * sigma2_t\
            + (x - beta) ** 2 * s_v_x\
            + b_dot_0 ** 2 * (sigma2_t + alpha ** 2 * s_v_x) \
            - 2 * b_dot_0 * (mu_v_x * sigma2_t + alpha * (x - beta) * s_v_x)
    exparg = -exparg / (2 * sigma2_t * s_v_x)

    # move outer exponential inside paranthesis to cancel out exp(q) so it won't explode

    erfterm1 = np.exp(exparg) / np.sqrt(np.pi)
    erfterm2 =  q * np.exp(q ** 2 + exparg) * (1 + erf(q))
    erfterm = erfterm1 + erfterm2


    prefactor = np.sqrt(np.pi) * (s_v_x * sigma2_t) /  (alpha ** 2 * s_v_x + sigma2_t)

    # normalizing factors
    f1 = 1. / integral_f1_xdot(b_0, b_dot_0, mu_0, s_0)  # prefactor from distribution p(v) at time 0
    n1 = 1. / np.sqrt(2 * np.pi * sigma2_t)  # prefactor from normal dist. p(x) at time t
    n2 = 1. / np.sqrt(2 * np.pi * s_v_x)     # prefactor from normal dist p(v|x=b) at time 0
    
    return f1 * n1 * n2 * prefactor * erfterm

@ignore_numpy_warnings
def ou_soln_marginal_v_after_upcrossing(x, mu_0, s_0, b_0, b_dot_0, t, p):
    alpha, beta = ou_soln_upcrossing_alpha_beta(t, p)
    alpha = alpha[0]
    beta = beta[0]
    S = ou_soln_upcrossing_S(t, p)
    sigma2_t = S[0,0]
    beta *= b_0

    mu_v_x, s_v_x = conditional_bivariate_gaussian(b_0, mu_0, s_0) 

    q = (mu_v_x * sigma2_t + alpha * (x - beta) * s_v_x - b_dot_0 * (sigma2_t + alpha ** 2 * s_v_x))\
             / np.sqrt(2 * sigma2_t * s_v_x * (alpha ** 2 * s_v_x + sigma2_t))

    exparg = mu_v_x ** 2 * sigma2_t\
            + (x - beta) ** 2 * s_v_x\
            + b_dot_0 ** 2 * (sigma2_t + alpha ** 2 * s_v_x) \
            - 2 * b_dot_0 * (mu_v_x * sigma2_t + alpha * (x - beta) * s_v_x)
    exparg = -exparg / (2 * sigma2_t * s_v_x)

    erfterm = 1 / np.sqrt(np.pi) + q * np.exp(q ** 2) * (1 + erf(q))

    prefactor = np.sqrt(np.pi) * (s_v_x * sigma2_t) /  (alpha ** 2 * s_v_x + sigma2_t)

    # normalizing factors
    f1 = 1. / integral_f1_xdot(b_0, b_dot_0, mu_0, s_0)  # prefactor from distribution p(v) at time 0
    n1 = 1. / np.sqrt(2 * np.pi * sigma2_t)  # prefactor from normal dist. p(x) at time t
    n2 = 1. / np.sqrt(2 * np.pi * s_v_x)     # prefactor from normal dist p(v|x=b) at time 0
 


    return f1 * n1 * n2 * prefactor * np.exp(exparg) * erfterm







