import numpy as np
import scipy.optimize as opt
import pickle
import demographics_india as demog
import matplotlib.pyplot as plt

# define parameters
econ_life = 80  # number of years from begin work until death
frac_work = 4 / 5  # fraction of time from begin work until death
# that model agent is working
E = 20  # Number of non-economically relevant periods
S = 80  # Number of economically relevant periods
T = int(3 * S)  # Number of periods of time to simulate
min_yr = 1
max_yr = 100
curr_year = 2018

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

g_y = 0.04

'''
Read demographic parameters
This section calls demographics_india.get_pop_obs(). This function takes
the arguments listed as inputs, and outputs the tuple of arguments that
is the time path of the stationary population distribution, the steady-
state population growth rate, the steady-state population distribution,
mortality rates, the time path of growth rates, the time path of
immigration rates, the population distribution by age in the period
immediately preceding the first period.
'''
# demog_obs = pickle.load(open('demographic_objects.pkl', 'rb'))
(omega_path, g_n_ss, omega_ss, mort_rates, g_n_path, imm_rates_path,
    omega_S_preTP) = demog.get_pop_objs(E, S, T, min_yr, max_yr,
                                        curr_year, GraphDiag=False)
imm_rates = imm_rates_path[-1, :]


def get_BQ(bvec, r, g_n, omega, mort_rates):
    BQ = ((1 + r) / (1 + g_n)) * (mort_rates[:-1] * omega[:-1] *
                                  bvec).sum()
    return BQ


def get_K(bvec, g_n, omega, imm_rates):
    '''
    Compute the aggregate capital stock
    '''
    K = ((1 / (1 + g_n)) *
         (omega[:-1] * bvec + imm_rates[1:] * omega[1:] * bvec).sum())
    return K


def get_L(nvec, omega):
    '''
    Compute aggregate labor
    '''
    L = (omega * nvec).sum()
    return L


def get_C_aggr(cvec, omega):
    '''
    Compute aggregate labor
    '''
    C = (omega * cvec).sum()
    return C


def get_Y(K, L, A, alpha):
    Y = A * (K ** alpha) * (L ** (1 - alpha))
    return Y


def get_Iss(K, g_n, g_y, delta):
    Iss = K * (np.exp(g_y) * (1 + g_n) - (1 - delta))
    return Iss


def get_NXss(bvec, omega, imm_rates, g_n, g_y):
    NX = -np.exp(g_y) * (imm_rates[1:] * omega[1:] * bvec).sum()
    return NX


def get_r(Y, K, A, alpha, delta):
    r = alpha * (Y / K) - delta
    return r


def get_w(Y, L, A, alpha):
    w = (1 - alpha) * (Y / L)
    return w


def get_c(nvec, bvec, r, w, g_y, BQ):
    b_s = np.append(0.0, bvec)
    b_splus1 = np.append(bvec, 0.0)
    cvec = (1 + r) * b_s + w * nvec + BQ - np.exp(g_y) * b_splus1
    return cvec


def u_prime(cvec, sigma):
    MUc = cvec ** (-sigma)
    return MUc


def euler_system(bvec, *args):
    (nvec, beta, sigma, A, alpha, delta, omega, mort_rates, imm_rates,
        g_n, g_y) = args
    K = get_K(bvec, g_n, omega, imm_rates)
    L = get_L(nvec, omega)
    Y = get_Y(K, L, A, alpha)
    r = get_r(Y, K, A, alpha, delta)
    w = get_w(Y, L, A, alpha)
    BQ = get_BQ(bvec, r, g_n, omega, mort_rates)
    cvec = get_c(nvec, bvec, r, w, g_y, BQ)
    euler_errors = (u_prime(cvec[:-1], sigma) -
                    np.exp(-sigma * g_y) * beta *
                    (1 - mort_rates[:-1]) *
                    (1 + r) * u_prime(cvec[1:], sigma))
    return euler_errors


bvec_init = np.ones(S - 1) * 0.01
eul_args = (nvec, beta, sigma, A, alpha, delta, omega_ss, mort_rates,
            imm_rates, g_n_ss, g_y)
results = opt.root(euler_system, bvec_init, args=(eul_args), tol=1e-14)
b_ss = results.x
b_errors = results.fun
print('The SS savings are: ', b_ss)
print('The maximum absolute Euler error is: ',
      np.absolute(b_errors).max())

K_ss = get_K(b_ss, g_n_ss, omega_ss, imm_rates)
L_ss = get_L(nvec, omega_ss)
Y_ss = get_Y(K_ss, L_ss, A, alpha)
r_ss = get_r(Y_ss, K_ss, A, alpha, delta)
w_ss = get_w(Y_ss, L_ss, A, alpha)
BQ_ss = get_BQ(b_ss, r_ss, g_n_ss, omega_ss, mort_rates)
c_ss = get_c(nvec, b_ss, r_ss, w_ss, g_y, BQ_ss)
Caggr_ss = get_C_aggr(c_ss, omega_ss)
I_ss = get_Iss(K_ss, g_n_ss, g_y, delta)
NX_ss = get_NXss(b_ss, omega_ss, imm_rates, g_n_ss, g_y)
RC_error = Y_ss - Caggr_ss - I_ss - NX_ss

print('The SS resource constraint is: ', RC_error)

print('r_ss=', r_ss, ', w_ss=', w_ss, ', K_ss=', K_ss, ', L_ss=', L_ss,
      ', Y_ss=', Y_ss, ', BQ_ss=', BQ_ss)

plt.plot(b_ss)
# plt.show()
plt.close()
