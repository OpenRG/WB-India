'''
------------------------------------------------------------------------
This program runs the steady state solver as well as the time path
iteration solution for the model with S-period lived agents, endogenous
labor, and heterogeneous abilities from Chapter 8 of the OG textbook.

This Python script imports the following module(s):
    SS.py
    TPI.py
    aggregates.py
    ability.py
    elliputil.py
    utilities.py

This Python script calls the following function(s):
    abil.get_e_interp()
    elp.fit_ellip_CFE()
    ss.get_SS_bsct()
    utils.compare_args()
    aggr.get_K()
    tpi.get_TPI()

Files created by this script:
    OUTPUT/SS/ss_vars.pkl
    OUTPUT/SS/ss_args.pkl
    OUTPUT/TPI/tpi_vars.pkl
    OUTPUT/TPI/tpi_args.pkl
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import pickle
import os
import SS as ss
import TPI as tpi
import aggregates as aggr
import ability as abil
import elliputil as elp
import utilities as utils

'''
------------------------------------------------------------------------
Declare parameters
------------------------------------------------------------------------
S             = integer in [3,80], number of periods an individual lives
lambdas       = (J,) vector, income percentiles for distribution of
                ability within each cohort
err_msg       = string, error message
J             = integer >= 1, number of heterogeneous ability groups
beta_annual   = scalar in (0,1), discount factor for one year
beta          = scalar in (0,1), discount factor for each model period
sigma         = scalar > 0, coefficient of relative risk aversion
l_tilde       = scalar > 0, per-period time endowment for every agent
chi_n_vec     = (S,) vector, values for chi^n_s
start_age     = integer >= 0, beginning age in years at which agents are
                born. For example, start_age = 0 means agents are born
                at the beginning of their 0th year (true day of birth)
end_age       = integer > start_age, year of life at the end of which
                agents die with certainty. end_age = 100 means agents
                die at the end of their 100th year
mod_age_dist  = (S,) vector, population distribution by model age
dat_age_dist  = (end_age-start_age+1,) vector, data population
                distribution by age
emat          = (S, J) matrix, e_{j,s} ability by age and income group
A             = scalar > 0, total factor productivity parameter in
                firms' production function
alpha         = scalar in (0,1), capital share of income
delta_annual  = scalar in [0,1], one-year depreciation rate of capital
delta         = scalar in [0,1], model-period depreciation rate of
                capital
SS_solve      = boolean, =True if want to solve for steady-state
                solution, otherwise retrieve solutions from pickle
SS_tol        = scalar > 0, tolerance level for steady-state fsolve
SS_graphs     = boolean, =True if want graphs of steady-state objects
SS_EulDiff    = boolean, =True if use simple differences in Euler
                errors. Otherwise, use percent deviation form.
T1            = integer > S, number of time periods until steady state
                is assumed to be reached
T2            = integer > T1, number of time periods after which steady-
                state is forced in TPI
TPI_solve     = boolean, =True if want to solve TPI after solving SS
TPI_tol       = scalar > 0, tolerance level for fsolve's in TPI
maxiter_TPI   = integer >= 1, Maximum number of iterations for TPI
mindist_TPI   = scalar > 0, Convergence criterion for TPI
xi_TPI        = scalar in (0,1], TPI path updating parameter
TPI_graphs    = Boolean, =True if want graphs of TPI objects
TPI_EulDiff   = Boolean, =True if want difference version of Euler
                errors beta*(1+r)*u'(c2) - u'(c1), =False if want ratio
                version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
------------------------------------------------------------------------
'''
# Household parameters
S = int(80)
lambdas = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])
# Make sure that lambdas vector sums to 1.0
if not np.isclose(1.0, lambdas.sum()):
    err_msg = ('ERROR: lambdas vector does not sum to one.')
    raise RuntimeError(err_msg)
J = lambdas.shape[0]
beta_annual = 0.96
beta = beta_annual ** (80 / S)
sigma = 2.5
l_tilde = 1.0
chi_n_vec = 1.0 * np.ones(S)
start_age = 21
end_age = 100
mod_age_dist = (1 / S) * np.ones(S)
dat_age_dist = ((1 / (end_age - start_age + 1)) *
                np.ones(end_age - start_age + 1))
emat = abil.get_e_interp(S, mod_age_dist, dat_age_dist, lambdas,
                         plot=False)
# Firm parameters
A = 1.0
alpha = 0.35
delta_annual = 0.05
delta = 1 - ((1 - delta_annual) ** (80 / S))
# SS parameters
SS_solve = True
SS_tol = 1e-13
SS_graphs = True
SS_EulDiff = True
# TPI parameters
T1 = int(round(3.0 * S))
T2 = int(round(3.5 * S))
TPI_solve = True
TPI_tol = 1e-13
maxiter_TPI = 200
mindist_TPI = 1e-7
xi_TPI = 0.20
TPI_graphs = True
TPI_EulDiff = True

'''
------------------------------------------------------------------------
Fit elliptical utility function to constant Frisch elasticity (CFE)
disutility of labor function by matching marginal utilities along the
support of leisure
------------------------------------------------------------------------
ellip_graph  = Boolean, =True if want to save plot of fit
b_ellip_init = scalar > 0, initial guess for b
upsilon_init = scalar > 1, initial guess for upsilon
ellip_init   = (2,) vector, initial guesses for b and upsilon
Frisch_elast = scalar > 0, Frisch elasticity of labor supply for CFE
               disutility of labor
CFE_scale    = scalar > 0, scale parameter for CFE disutility of labor
cfe_params   = (2,) vector, values for (Frisch, CFE_scale)
b_ellip      = scalar > 0, fitted value of b for elliptical disutility
               of labor
upsilon      = scalar > 1, fitted value of upsilon for elliptical
               disutility of labor
------------------------------------------------------------------------
'''
ellip_graph = False
b_ellip_init = 1.0
upsilon_init = 2.0
ellip_init = np.array([b_ellip_init, upsilon_init])
Frisch_elast = 0.8
CFE_scale = 1.0
cfe_params = np.array([Frisch_elast, CFE_scale])
b_ellip, upsilon = elp.fit_ellip_CFE(ellip_init, cfe_params, l_tilde,
                                     ellip_graph)

'''
------------------------------------------------------------------------
Solve for the steady-state solution
------------------------------------------------------------------------
cur_path       = string, current file path of this script
ss_output_fldr = string, cur_path extension of SS output folder path
ss_output_dir  = string, full path name of SS output folder
ss_outputfile  = string, path name of file for SS output objects
ss_paramsfile  = string, path name of file for SS parameter objects
rss_init       = scalar > 0, initial guess for r_ss
c1_init        = scalar > 0, initial guess for c1
init_vals      = length 3 tuple, initial values to be passed in to
                 get_SS_bsct()
ss_args        = length 15 tuple, args to be passed in to get_SS_bsct()
ss_output      = length 14 dict, steady-state objects {n_ss, b_ss, c_ss,
                 b_Sp1_ss, w_ss, r_ss, K_ss, L_ss, Y_ss, C_ss, n_err_ss,
                 b_err_ss, RCerr_ss, ss_time}
ss_vars_exst   = boolean, =True if ss_vars.pkl exists
ss_args_exst   = boolean, =True if ss_args.pkl exists
err_msg        = string, error message
cur_ss_args    = length 17 tuple, current args
args_same      = boolean, =True if ss_args == cur_ss_args
------------------------------------------------------------------------
'''
# Create OUTPUT/SS directory if does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
ss_output_fldr = 'OUTPUT/SS'
ss_output_dir = os.path.join(cur_path, ss_output_fldr)
if not os.access(ss_output_dir, os.F_OK):
    os.makedirs(ss_output_dir)
ss_outputfile = os.path.join(ss_output_dir, 'ss_vars.pkl')
ss_paramsfile = os.path.join(ss_output_dir, 'ss_args.pkl')

# Compute steady-state solution
if SS_solve:
    print('BEGIN EQUILIBRIUM STEADY-STATE COMPUTATION')
    rss_init = 0.10
    c1_init = 0.03
    init_vals = (rss_init, c1_init)
    ss_args = (J, S, lambdas, emat, beta, sigma, l_tilde, b_ellip,
               upsilon, chi_n_vec, A, alpha, delta, SS_tol, SS_EulDiff)
    print('Solving SS outer loop using bisection method.')
    ss_output = ss.get_SS_bsct(init_vals, ss_args, SS_graphs)

    # Save ss_output as pickle
    pickle.dump(ss_output, open(ss_outputfile, 'wb'))
    pickle.dump(ss_args, open(ss_paramsfile, 'wb'))

# Don't compute steady-state, get it from pickle
else:
    # Make sure that the SS output files exist
    ss_vars_exst = os.path.exists(ss_outputfile)
    ss_args_exst = os.path.exists(ss_paramsfile)
    if (not ss_vars_exst) or (not ss_args_exst):
        # If the files don't exist, stop the program and run the steady-
        # state solution first
        err_msg = ('ERROR: The SS output files do not exist and ' +
                   'SS_solve=False. Must set SS_solve=True and ' +
                   'compute steady-state solution.')
        raise RuntimeError(err_msg)
    else:
        # If the files do exist, make sure that none of the parameters
        # changed from the parameters used in the solution for the saved
        # steady-state pickle
        ss_args = pickle.load(open(ss_paramsfile, 'rb'))
        cur_ss_args = (J, S, lambdas, emat, beta, sigma, l_tilde,
                       b_ellip, upsilon, chi_n_vec, A, alpha, delta,
                       SS_tol, SS_EulDiff)
        args_same = utils.compare_args(ss_args, cur_ss_args)
        if args_same:
            # If none of the parameters changed, use saved pickle
            print('RETRIEVE STEADY-STATE SOLUTIONS FROM FILE')
            ss_output = pickle.load(open(ss_outputfile, 'rb'))
        else:
            # If any of the parameters changed, end the program and
            # compute the steady-state solution
            err_msg = ('ERROR: Current ss_args are not equal to the ' +
                       'ss_args that produced ss_output. Must solve ' +
                       'for SS before solving transition path. Set ' +
                       'SS_solve=True.')
            raise RuntimeError(err_msg)

'''
------------------------------------------------------------------------
Solve for the transition path equilibrium by time path iteration (TPI)
------------------------------------------------------------------------
tpi_output_fldr = string, cur_path extension of TPI output folder path
tpi_output_dir  = string, full path name of TPI output folder
tpi_outputfile  = string, path name of file for TPI output objects
tpi_paramsfile  = string, path name of file for TPI parameter objects
K_ss            = scalar > 0, steady-state aggregate capital stock
L_ss            = scalar > 0, steady-state aggregate labor
C_ss            = scalar > 0, steady-state aggregate consumption
b_ss            = (S, J) matrix, steady-state savings distribution
init_wgts       = (S, J) matrix, weights representing the factor by
                  which the initial wealth distribution differs from the
                  steady-state wealth distribution
bvec1           = (S,) vector, initial period savings distribution
K1              = scalar, initial period aggregate capital stock
K1_cstr         = Boolean, =True if K1 <= 0
tpi_params      = length 23 tuple, args to pass into tpi.get_TPI()
tpi_output      = length 14 dictionary, {cpath, npath, bpath, wpath,
                  rpath, Kpath, Lpath, Ypath, Cpath, bSp1_err_path,
                  b_err_path, n_err_path, RCerrPath, tpi_time}
tpi_args        = length 24 tuple, args that were passed in to get_TPI()
------------------------------------------------------------------------
'''
if TPI_solve:
    print('BEGIN EQUILIBRIUM TRANSITION PATH COMPUTATION')

    # Create OUTPUT/TPI directory if does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    tpi_output_fldr = 'OUTPUT/TPI'
    tpi_output_dir = os.path.join(cur_path, tpi_output_fldr)
    if not os.access(tpi_output_dir, os.F_OK):
        os.makedirs(tpi_output_dir)
    tpi_outputfile = os.path.join(tpi_output_dir, 'tpi_vars.pkl')
    tpi_paramsfile = os.path.join(tpi_output_dir, 'tpi_args.pkl')

    r_ss = ss_output['r_ss']
    K_ss = ss_output['K_ss']
    L_ss = ss_output['L_ss']
    C_ss = ss_output['C_ss']
    b_ss = ss_output['b_ss']
    n_ss = ss_output['n_ss']

    # Choose initial period distribution of wealth (bmat1), which
    # determines initial period aggregate capital stock
    init_wgts = 0.95 * np.ones((S, J))
    bmat1 = init_wgts * b_ss

    # Make sure init. period distribution is feasible in terms of K
    K1, K1_cstr = aggr.get_K(bmat1, lambdas)

    # If initial bvec1 is not feasible end program
    if K1_cstr:
        print('Initial savings distribution is not feasible because ' +
              'K1<epsilon. Some element(s) of bmat1 must increase.')
    else:
        tpi_params = (J, S, T1, T2, lambdas, emat, beta, sigma, l_tilde,
                      b_ellip, upsilon, chi_n_vec, A, alpha, delta,
                      r_ss, K_ss, L_ss, C_ss, b_ss, n_ss, maxiter_TPI,
                      mindist_TPI, TPI_tol, xi_TPI, TPI_EulDiff)
        tpi_output = tpi.get_TPI(tpi_params, bmat1, TPI_graphs)

        tpi_args = (J, S, T1, T2, lambdas, emat, beta, sigma, l_tilde,
                    b_ellip, upsilon, chi_n_vec, A, alpha, delta, r_ss,
                    K_ss, L_ss, C_ss, b_ss, n_ss, maxiter_TPI,
                    mindist_TPI, TPI_tol, xi_TPI, TPI_EulDiff, bmat1)

        # Save tpi_output as pickle
        pickle.dump(tpi_output, open(tpi_outputfile, 'wb'))
        pickle.dump(tpi_args, open(tpi_paramsfile, 'wb'))
