import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# define parameters
econ_life = 50  # number of years from begin work until death
frac_work = 4 / 5  # fraction of time from begin work until death
# that model agent is working
S = 50  # number of model periods
alpha = 0.3
A = 1
delta_annual = 0.05
delta = 1 - ((1 - delta_annual) ** (econ_life / S))
beta_annual = 0.96
beta = beta_annual ** (econ_life / S)
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


def lone_euler(b32, *args):
    (r1, r2, w1, w2, nvec, b21, beta, sigma) = args
    c21 = (1 + r1) * b21 + w1 * nvec[1] - b32
    c32 = (1 + r2) * b32 + w2 * nvec[2]
    euler_error = (u_prime(c21, sigma) - beta * (1 + r2) *
                   u_prime(c32, sigma))
    return euler_error


def euler_sys_tpi(guesses, *args):
    bvec = guesses
    (rpath, wpath, nvec, b_init, beta, sigma) = args
    b_s_vec = np.append(b_init, bvec)
    b_sp1_vec = np.append(bvec, 0.0)
    cvec = wpath * nvec + (1 + rpath) * b_s_vec - b_sp1_vec
    euler_errors = (u_prime(cvec[:-1], sigma) - beta * (1 + rpath[1:]) *
                    u_prime(cvec[1:], sigma))
    return euler_errors


def get_r_path(L, K, A, alpha, delta):
    rpath = alpha * A * (L / K) ** (1 - alpha) - delta
    return rpath


def get_w_path(L, K, A, alpha):
    wpath = (1 - alpha) * A * (K / L) ** alpha
    return wpath


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
print('The SS interest rate is: ', r_ss)
print('The SS wage rate is: ', w_ss)
print('The SS capital stock is: ', K_ss)
print('The SS labor supply is: ', L_ss)


'''
Starting the time path solution
'''
bvec1 = 0.9 * b_ss
T = 3 * S
xi = 0.2
K1 = get_K(bvec1)
Kpath = np.linspace(K1, K_ss, num=T)
Kpath = np.append(Kpath, K_ss * np.ones(S-2))


dist = 10
tpi_iter = 0
tpi_max_iter = 100
tpi_tol = 1e-6
while (dist > tpi_tol) & (tpi_iter < tpi_max_iter):
    tpi_iter += 1
    L = get_L(nvec)
    rpath = get_r_path(L, Kpath, A, alpha, delta)
    wpath = get_w_path(L, Kpath, A, alpha)
    bmat = np.zeros((T + S - 2, S - 1))
    bmat[0, :] = bvec1
    for p in range(1, S - 1):
        bguess = np.diag(bmat[:p, -p:])
        b_args = (rpath[:p + 1], wpath[:p + 1], nvec[-p - 1:], bvec1[-p],
                  beta, sigma)
        results = opt.root(euler_sys_tpi, bguess, args=(b_args))
        bvec = results.x
        diagmask = np.eye(p, dtype=bool)
        bmat[1:p + 1, -p:] = diagmask * bvec + bmat[1:p + 1, -p:]

    for t in range(1, T):
        bguess = np.diag(bmat[t - 1: t + S - 1, :])
        b_args = (rpath[t - 1: t + S - 1], wpath[t - 1: t + S - 1],
                  nvec, 0.0, beta, sigma)
        results = opt.root(euler_sys_tpi, bguess, args=(b_args))
        bvec = results.x
        diagmask = np.eye(S - 1, dtype=bool)
        bmat[t:t + S - 1, :] = diagmask * bvec + bmat[t:t + S - 1, :]

    Kprime = bmat.sum(axis=1)
    dist = ((Kpath[:T] - Kprime[:T]) ** 2).sum()
    print('Distance at iteration ', tpi_iter, ' is ', dist)
    Kpath[:T] = xi * Kprime[:T] + (1 - xi) * Kpath[:T]

print('Kpath = ', Kpath)
plt.plot(rpath)
plt.show()
