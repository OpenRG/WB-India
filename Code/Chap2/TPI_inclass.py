import numpy as np
import scipy.optimize as opt

# define parameters
alpha = 0.3
A = 1
delta_annual = 0.05
# supposing a model period represents 20 years
delta = 1 - ((1 - delta_annual) ** 20)
beta_annual = 0.96
beta = beta_annual ** 20  # supposing a model period represents 20 years
sigma = 2.0
nvec = np.array([1.0, 1.0, 0.2])


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
    euler_errors = (u_prime(cvec[:2], sigma) - beta * (1+r) *
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
    b2, b3 = guesses
    (r_tp1, r_tp2, w_t, w_tp1, w_tp2, nvec, beta, sigma) = args
    c1 = w_t * nvec[0] - b2
    c2 = (1 + r_tp1) * b2 + w_tp1 * nvec[1] - b3
    c3 = (1 + r_tp2) * b3 + w_tp2 * nvec[2]
    euler_error1 = (u_prime(c1, sigma) - beta * (1 + r_tp1) *
                    u_prime(c2, sigma))
    euler_error2 = (u_prime(c2, sigma) - beta * (1 + r_tp2) *
                    u_prime(c3, sigma))
    euler_errors = [euler_error1, euler_error2]
    return euler_errors


def get_r_path(L, K, A, alpha, delta):
    rpath = alpha * A * (L / K) ** (1 - alpha) - delta
    return rpath


def get_w_path(L, K, A, alpha):
    wpath = (1 - alpha) * A * (K / L) ** alpha
    return wpath


bvec_init = np.array([0.01, 0.4])
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
      (1 + r_ss) ** (1 / 20) - 1)
print('The SS wage rate is: ', w_ss)
print('The SS capital stock is: ', K_ss)
print('The SS labor supply is: ', L_ss)


'''
Starting the time path solution
'''
bvec1 = np.array([0.9 * b_ss[0], 1.2 * b_ss[1]])
T = 50
xi = 0.2
K1 = get_K(bvec1)
Kpath = np.linspace(K1, K_ss, num=T)

dist = 10
tpi_iter = 0
tpi_max_iter = 100
tpi_tol = 1e-6
while (dist > tpi_tol) & (tpi_iter < tpi_max_iter):
    tpi_iter += 1
    L = get_L(nvec)
    rpath = get_r_path(L, Kpath, A, alpha, delta)
    rpath = np.append(rpath, r_ss)
    wpath = get_w_path(L, Kpath, A, alpha)
    wpath = np.append(wpath, w_ss)
    b32_guess = bvec1[1]
    b32_args = (rpath[0], rpath[1], wpath[0], wpath[1], nvec, bvec1[0],
                beta, sigma)
    results = opt.root(lone_euler, b32_guess, args=(b32_args))
    b32 = results.x

    bmat = np.zeros((T + 1, 2))
    bmat[0, :] = bvec1
    bmat[1, 1] = b32
    for t in range(1, T - 1):
        bguess = [bmat[t-1, 0], bmat[t, 1]]
        b_args = (rpath[t+1], rpath[t+2], wpath[t], wpath[t+1],
                  wpath[t+2], nvec, beta, sigma)
        results = opt.root(euler_sys_tpi, bguess, args=(b_args))
        b2, b3 = results.x
        bmat[t, 0] = b2
        bmat[t+1, 1] = b3

    Kprime = bmat.sum(axis=1)
    dist = ((Kpath[:-1] - Kprime[:-2]) ** 2).sum()
    print('Distance at iteration ', tpi_iter, ' is ', dist)
    Kpath = xi * Kprime[:-2] + (1 - xi) * Kpath[:-1]
    Kpath = np.append(Kpath, K_ss)

print('Kpath = ', Kpath)
import matplotlib.pyplot as plt

plt.plot(Kpath)
plt.show()
