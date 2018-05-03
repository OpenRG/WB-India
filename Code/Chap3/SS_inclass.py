import numpy as np
import scipy.optimize as opt

# define parameters
econ_life = 50  # number of years from begin work until death
frac_work = 4 / 5  # fraction of time from begin work until death
# that model agent is working
S = 50  # number of model periods
alpha = 0.3
A = 1
delta_annual = 0.05
delta = 1 - ((1 - delta_annual) ** np.round(econ_life / S))
beta_annual = 0.96
beta = beta_annual ** np.round(econ_life / S)
sigma = 2.0
nvec = np.empty(S)
retire = int(np.round(frac_work * S)) + 1
nvec[:retire] = 1.0
nvec[retire:] = 0.2


def get_K(bvec):
    '''
    Compute the aggregate capital stock
    '''
    K = bvec.sum()
    return K


def get_L(nvec):
    '''
    Compute aggregate labor
    '''
    L = nvec.sum()

    return L


def get_r(nvec, bvec, A, alpha, delta):
    K = get_K(bvec)
    L = get_L(nvec)
    r = alpha * A * (L / K) ** (1 - alpha) - delta
    return r


def get_w(nvec, bvec, A, alpha):
    K = get_K(bvec)
    L = get_L(nvec)
    w = (1 - alpha) * A * (K / L) ** alpha
    return w


def get_c(nvec, bvec, r, w):
    b_s = np.append(0.0, bvec)
    b_splus1 = np.append(bvec, 0.0)
    cvec = (1 + r) * b_s + w * nvec - b_splus1
    return cvec


def u_prime(cvec, sigma):
    MUc = cvec ** -sigma
    return MUc


def euler_system(bvec, *args):
    (nvec, beta, sigma, A, alpha, delta) = args
    r = get_r(nvec, bvec, A, alpha, delta)
    w = get_w(nvec, bvec, A, alpha)
    cvec = get_c(nvec, bvec, r, w)
    euler_errors = (u_prime(cvec[:-1], sigma) - beta * (1+r) *
                    u_prime(cvec[1:], sigma))
    return euler_errors


bvec_init = np.ones((S - 1)) * 0.01
eul_args = (nvec, beta, sigma, A, alpha, delta)
results = opt.root(euler_system, bvec_init, args=(eul_args), tol=1e-14)
b_ss = results.x
zero_val = results.fun
print('The SS savings are: ', b_ss, ' the errors are: ', zero_val)
r_ss = get_r(nvec, b_ss, A, alpha, delta)
w_ss = get_w(nvec, b_ss, A, alpha)
K_ss = get_K(b_ss)
L_ss = get_L(nvec)
print('The SS interest rate is: ', r_ss, ' Annual rate of: ',
      (1 + r_ss) ** (1 / (econ_life / S)) - 1)
print('The SS wage rate is: ', w_ss)
print('The SS capital stock is: ', K_ss)
print('The SS labor supply is: ', L_ss)

import matplotlib.pyplot as plt
plt.plot(b_ss)
plt.show()
