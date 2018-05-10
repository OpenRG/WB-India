import numpy as np
import os
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# Define parameters
econ_life = 50  # years households are economically active real life
S = 50  # number of model periods

alpha = 0.3
A = 1
delta_annual = 0.05
delta = 1 - ((1 - delta_annual) ** np.round(econ_life / S))

beta_annual = 0.96
beta = beta_annual ** np.round(econ_life / S)
sigma = 2.0
l_tilde = 1.0
b_ellip = 0.629
upsilon = 1.753
chi_n_vec = np.ones(S)

# Tax parameters
ETR = 0.24
MTRx = 0.35
MTRy = 0.01


# Define functions


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


def get_r(K, L, A, alpha, delta):
    r = alpha * A * (L / K) ** (1 - alpha) - delta
    return r


def get_w_from_r(r, A, alpha, delta):
    w = (1 - alpha) * A * ((alpha * A) / (r + delta)) ** (alpha /
                                                          (1 - alpha))
    return w


def get_Y(K, L, A, alpha):
    Y = A * (K ** alpha) * (L ** (1 - alpha))
    return Y


def get_TaxLiab(nvec, bvec, r, w, ETR):
    b_s = np.append(0.0, bvec)
    TaxLiab = ETR * (r * b_s + w * nvec)
    return TaxLiab


def get_c(nvec, bvec, r, w, X, ETR):
    b_s = np.append(0.0, bvec)
    b_splus1 = np.append(bvec, 0.0)
    TaxLiab = get_TaxLiab(nvec, bvec, r, w, ETR)
    cvec = (1 + r) * b_s + w * nvec + X - TaxLiab - b_splus1
    return cvec


def MDU_n(nvec, l_tilde, b_ellip, upsilon):
    MDU = ((b_ellip / l_tilde) * ((nvec / l_tilde) ** (upsilon - 1)) *
           ((1 - ((nvec / l_tilde) ** upsilon)) ** ((1 - upsilon) /
                                                    upsilon)))
    return MDU


def MU_c(cvec, sigma):
    MUc = cvec ** -sigma
    return MUc


def get_n_errors(nvec, bvec, r, w, X, args):
    sigma, chi_n_vec, l_tilde, b_ellip, upsilon, ETR, MTRx = args
    MDU_ns = MDU_n(nvec, l_tilde, b_ellip, upsilon)
    c_s = get_c(nvec, bvec, r, w, X, ETR)
    MU_cs = MU_c(c_s, sigma)
    n_errors = w * (1 - MTRx) * MU_cs - chi_n_vec * MDU_ns
    return n_errors


def get_b_errors(nvec, bvec, r, w, X, args):
    beta, sigma, ETR, MTRy = args
    cvec = get_c(nvec, bvec, r, w, X, ETR)
    c_s = cvec[:-1]
    c_sp1 = cvec[1:]
    MU_cs = MU_c(c_s, sigma)
    MU_csp1 = MU_c(c_sp1, sigma)
    b_errors = MU_cs - beta * (1 + r * (1 - MTRy)) * MU_csp1
    return b_errors


def euler_system(nbvec, *args):
    (r, w, X, ETR, MTRx, MTRy, beta, sigma, chi_n_vec, l_tilde, b_ellip,
        upsilon, S) = args
    nvec = nbvec[:S]
    bvec = nbvec[S:]
    n_args = (sigma, chi_n_vec, l_tilde, b_ellip, upsilon, ETR, MTRx)
    n_errors = get_n_errors(nvec, bvec, r, w, X, n_args)
    b_args = (beta, sigma, ETR, MTRy)
    b_errors = get_b_errors(nvec, bvec, r, w, X, b_args)
    nb_errors = np.append(n_errors, b_errors)
    return nb_errors

r_init = 0.10
X_init = 0.10
rX_init = np.array([r_init, X_init])

ss_dist = 10.0
ss_out_tol = 1e-14
ss_maxiter = 200
ss_iter = 0
xi_ss = 0.4
while (ss_dist > ss_out_tol) & (ss_iter < ss_maxiter):
    ss_iter += 1
    r, X = rX_init

    w = get_w_from_r(r, A, alpha, delta)
    nvec_init = 0.9 * np.ones(S)
    bvec_init = 0.05 * np.ones(S - 1)
    nbvec_init = np.append(nvec_init, bvec_init)
    nb_args = (r, w, X_init, ETR, MTRx, MTRy, beta, sigma, chi_n_vec,
               l_tilde, b_ellip, upsilon, S)
    results = opt.root(euler_system, nbvec_init, args=(nb_args),
                       method='lm', tol=1e-14)
    n_new = results.x[:S]
    b_new = results.x[S:]
    n_errors_new = results.fun[:S]
    b_errors_new = results.fun[:S]

    K = get_K(b_new)
    L = get_L(n_new)
    r_new = get_r(K, L, A, alpha, delta)
    w_new = get_w_from_r(r_new, A, alpha, delta)
    Rev_new = ETR * (r_new * np.append(0, b_new) + w_new * n_new).sum()
    X_new = Rev_new / S
    rX_new = np.array([r_new, X_new])
    ss_dist = np.absolute(rX_new - rX_init).max()
    print('SS iteration', ss_iter, ', distance=', ss_dist)
    rX_init = xi_ss * rX_new + (1 - xi_ss) * rX_init

b_ss = b_new
n_ss = n_new
n_errors_ss = n_errors_new
b_errors_ss = b_errors_new
X_ss = X_new
K_ss = get_K(b_ss)
L_ss = get_L(n_ss)
r_ss = get_r(K_ss, L_ss, A, alpha, delta)
w_ss = get_w_from_r(r_ss, A, alpha, delta)
Rev_ss = ETR * (r_ss * np.append(0, b_ss) + w_ss * n_ss).sum()
X_ss = Rev_ss / S
c_ss = get_c(n_ss, b_ss, r_ss, w_ss, X_ss, ETR)
C_ss = c_ss.sum()

Y_ss = get_Y(K_ss, L_ss, A, alpha)

res_cnstr_errors = Y_ss - C_ss - delta * K_ss

print('n_ss=', n_ss)
print('b_ss=', b_ss)
print('Max. abs. n_error=', np.absolute(n_errors_ss).max())
print('Max. abs. b_error=', np.absolute(b_errors_ss).max())
print('Max. abs. res. cnstr. error=',
      np.absolute(res_cnstr_errors).max())
print('r_ss=', r_ss, ', w_ss=', w_ss, ', K_ss=', K_ss, ', L_ss=', L_ss,
      ', Y_ss=', Y_ss, ', X_ss=', X_ss, ', Rev_ss=', Rev_ss)

'''
------------------------------------------------------------------------
Plot steady-state consumption, savings, and labor distributions
------------------------------------------------------------------------
'''

# Create directory if images directory does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)


# Plot steady-state consumption and labor distributions
b_ss_full = np.append(0, b_ss)
b_ss_full = np.append(b_ss_full, 0)
age_pers_c = np.arange(21, 21 + S)
age_pers_b = np.arange(21, 21 + S + 1)
fig, ax = plt.subplots()
plt.plot(age_pers_c, c_ss, marker='D', label='Consumption')
plt.plot(age_pers_b, b_ss_full, marker='D', label='Savings')
# for the minor ticks, use no labels; default NullFormatter
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
# plt.title('Steady-state consumption and savings', fontsize=20)
plt.xlabel(r'Age $s$')
plt.ylabel(r'Units of consumption')
plt.xlim((20, 20 + S + 2))
# plt.ylim((-1.0, 1.15 * (b_ss.max())))
plt.legend(loc='upper left')
output_path = os.path.join(output_dir, 'SS_bc')
plt.savefig(output_path)
# plt.show()
plt.close()

# Plot steady-state labor supply distribution
fig, ax = plt.subplots()
plt.plot(age_pers_c, n_ss, marker='D', label='Labor supply')
# for the minor ticks, use no labels; default NullFormatter
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
# plt.title('Steady-state labor supply', fontsize=20)
plt.xlabel(r'Age $s$')
plt.ylabel(r'Labor supply')
plt.xlim((20, 20 + S + 2))
# plt.ylim((-0.1, 1.15 * (n_ss.max())))
plt.legend(loc='upper right')
output_path = os.path.join(output_dir, 'SS_n')
plt.savefig(output_path)
# plt.show()
plt.close()
