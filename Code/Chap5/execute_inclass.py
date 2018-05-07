import numpy as np
import SSfuncs_inclass as SS

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
rbar = 0.10

J = 7
emat_full = np.genfromtxt('emat.csv', delimiter=',')
emat = emat_full[:, :J]
lambdas = np.array([0.25, 0.25, 0.20, 0.10, 0.10, 0.09, 0.01])

wbar = SS.get_w_from_r(rbar, A, alpha, delta)
nvec_init = 0.9 * np.ones(S)
bvec_init = 0.05 * np.ones(S - 1)
nbvec_init = np.append(nvec_init, bvec_init)
ss_args = (rbar, wbar, beta, sigma, chi_n_vec, l_tilde, b_ellip,
           upsilon, S, J, lambdas, emat, A, alpha, delta)
SS.SS_solve(nbvec_init, ss_args)
