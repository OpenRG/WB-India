'''
------------------------------------------------------------------------
This module contains the functions used to solve the steady state of
the model with S-period lived agents and exogenous labor supply from
Chapter 6 of the OG textbook.

This Python module imports the following module(s):
    households.py
    firms.py
    aggregates.py
    utilities.py

This Python module defines the following function(s):
    feasible()
    inner_loop()
    rw_errs()
    KL_errs()
    get_SS_root()
    get_SS_bsct()
    create_graphs()
------------------------------------------------------------------------
'''
# Import packages
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import households as hh
import firms
import aggregates as aggr
import utilities as utils

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def feasible(bvec, params):
    '''
    --------------------------------------------------------------------
    Check whether a vector of steady-state savings is feasible in that
    it satisfies the nonnegativity constraints on consumption in every
    period c_s > 0 and that the aggregate capital stock is strictly
    positive K > 0
    --------------------------------------------------------------------
    INPUTS:
    bvec   = (S-1,) vector, household savings b_{s+1}
    params = length 4 tuple, (nvec, A, alpha, delta)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_L()
        get_K()
        get_w()
        get_r()
        get_cvec()

    OBJECTS CREATED WITHIN FUNCTION:
    nvec     = (S,) vector, exogenous labor supply values n_s
    A        = scalar > 0, total factor productivity
    alpha    = scalar in (0, 1), capital share of income
    delta    = scalar in (0, 1), per-period depreciation rate
    S        = integer >= 3, number of periods in individual life
    L        = scalar > 0, aggregate labor
    K        = scalar, steady-state aggregate capital stock
    K_cstr   = boolean, =True if K <= 0
    w_params = length 2 tuple, (A, alpha)
    w        = scalar, steady-state wage
    r_params = length 3 tuple, (A, alpha, delta)
    r        = scalar, steady-state interest rate
    bvec2    = (S,) vector, steady-state savings distribution plus
               initial period wealth of zero
    cvec     = (S,) vector, steady-state consumption by age
    c_cnstr  = (S,) Boolean vector, =True for elements for which c_s<=0
    b_cnstr  = (S-1,) Boolean, =True for elements for which b_s causes a
               violation of the nonnegative consumption constraint

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_cnstr, c_cnstr, K_cnstr
    --------------------------------------------------------------------
    '''
    nvec, A, alpha, delta = params
    S = nvec.shape[0]
    L = aggr.get_L(nvec)
    K, K_cstr = aggr.get_K(bvec)
    if not K_cstr:
        w_params = (A, alpha)
        w = firms.get_w(K, L, w_params)
        r_params = (A, alpha, delta)
        r = firms.get_r(K, L, r_params)
        c_params = (nvec, r, w)
        cvec = hh.get_cons(bvec, 0.0, c_params)
        c_cstr = cvec <= 0
        b_cstr = c_cstr[:-1] + c_cstr[1:]

    else:
        c_cstr = np.ones(S, dtype=bool)
        b_cstr = np.ones(S - 1, dtype=bool)

    return c_cstr, K_cstr, b_cstr


def get_ss_graphs(c_ss, b_ss):
    '''
    --------------------------------------------------------------------
    Plot steady-state results
    --------------------------------------------------------------------
    INPUTS:
    c_ss = (S,) vector, steady-state lifetime consumption
    b_ss = (S-1,) vector, steady-state lifetime savings

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    S           = integer > 3, number of periods in lifetime
    bss2        = (S,) vector, [0, b_ss]
    age_pers    = (S,) vector, ages from 1 to S

    FILES CREATED BY THIS FUNCTION:
        SS_bc.png

    RETURNS: None
    ----------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    # Plot steady-state consumption and savings distributions
    S = c_ss.shape[0]
    b_ss2 = np.append(0.0, b_ss)
    age_pers = np.arange(1, S + 1)
    fig, ax = plt.subplots()
    plt.plot(age_pers, c_ss, marker='D', label='Consumption')
    plt.plot(age_pers, b_ss2, marker='D', label='Savings')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    # plt.title('Steady-state consumption and savings', fontsize=20)
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Units of consumption')
    plt.xlim((0, S + 1))
    # plt.ylim((-1.0, 1.15 * (b_ss.max())))
    plt.legend(loc='upper left')
    output_path = os.path.join(output_dir, 'SS_bc')
    plt.savefig(output_path)
    # plt.show()
    plt.close()


# def inner_loop(r, w, params):
#     '''
#     --------------------------------------------------------------------
#     Given values for r and w, solve for equilibrium errors from the two
#     first order conditions of the firm
#     --------------------------------------------------------------------
#     INPUTS:
#     params = length 14 tuple, (c1_init, S, beta, sigma, l_tilde, b_ellip,
#                             upsilon, chi_n_vec, A, alpha, delta, diff,
#                             hh_fsolve, SS_tol)
#     r      = scalar > 0, guess at steady-state interest rate
#     w      = scalar > 0, guess at steady-state wage

#     OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
#         hh.bn_solve()
#         hh.c1_bSp1err()
#         hh.get_cnb_vecs()
#         aggr.get_K()
#         aggr.get_L()
#         firms.get_r()
#         firms.get_w()

#     OBJECTS CREATED WITHIN FUNCTION:
#     c1_init    = scalar > 0, initial guess for c1
#     S          = integer >= 3, number of periods in individual lifetime
#     beta       = scalar in (0,1), discount factor
#     sigma      = scalar >= 1, coefficient of relative risk aversion
#     l_tilde    = scalar > 0, per-period time endowment for every agent
#     b_ellip    = scalar > 0, fitted value of b for elliptical disutility
#                  of labor
#     upsilon    = scalar > 1, fitted value of upsilon for elliptical
#                  disutility of labor
#     chi_n_vec  = (S,) vector, values for chi^n_s
#     A          = scalar > 0, total factor productivity parameter in
#                  firms' production function
#     alpha      = scalar in (0,1), capital share of income
#     delta      = scalar in [0,1], model-period depreciation rate of
#                  capital
#     diff       = boolean, =True if simple difference Euler errors,
#                  otherwise percent deviation Euler errors
#     hh_fsolve  = boolean, =True if solve inner-loop household problem by
#                  choosing c_1 to set final period savings b_{S+1}=0.
#                  Otherwise, solve the household problem as multivariate
#                  root finder with 2S-1 unknowns and equations
#     SS_tol     = scalar > 0, tolerance level for steady-state fsolve
#     rpath      = (S,) vector, lifetime path of interest rates
#     wpath      = (S,) vector, lifetime path of wages
#     c1_args    = length 10 tuple, args to pass into c1_bSp1err()
#     c1_options = length 1 dict, options for c1_bSp1err()
#     results_c1 = results object, results from c1_bSp1err()
#     c1         = scalar > 0, optimal initial period consumption given r
#                  and w
#     cnb_args   = length 8 tuple, args to pass into get_cnb_vecs()
#     cvec       = (S,) vector, lifetime consumption (c1, c2, ...cS)
#     nvec       = (S,) vector, lifetime labor supply (n1, n2, ...nS)
#     bvec       = (S,) vector, lifetime savings (b1, b2, ...bS) with b1=0
#     b_Sp1      = scalar, final period savings, should be close to zero
#     K          = scalar > 0, aggregate capital stock
#     K_cnstr    = boolean, =True if K < 0
#     L          = scalar > 0, aggregate labor
#     r_params   = length 3 tuple, (A, alpha, delta)
#     w_params   = length 2 tuple, (A, alpha)
#     r_new      = scalar > 0, guess at steady-state interest rate
#     w_new      = scalar > 0, guess at steady-state wage

#     FILES CREATED BY THIS FUNCTION: None

#     RETURNS: K, L, cvec, nvec, b_s_vec, b_splus1_vec, b_Sp1, r_new, w_new
#     --------------------------------------------------------------------
#     '''
#     c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon,\
#         chi_n_vec, A, alpha, delta, diff, hh_fsolve, SS_tol = params

#     if hh_fsolve:
#         b_init = np.ones((S - 1, 1)) * 0.05
#         n_init = np.ones((S, 1)) * 0.4
#         guesses = np.append(b_init, n_init)
#         bn_params = (r, w, S, beta, sigma, l_tilde, b_ellip,
#                      upsilon, chi_n_vec, diff)
#         [solutions, infodict, ier, message] = \
#             opt.fsolve(hh.bn_solve, guesses, args=bn_params,
#                        xtol=SS_tol, full_output=True)
#         euler_errors = infodict['fvec']
#         print('Max Euler errors: ', np.absolute(euler_errors).max())
#         b_splus1_vec = np.append(solutions[:S - 1], 0.0)
#         nvec = solutions[S - 1:]
#         b_s_vec = np.append(0.0, b_splus1_vec[:-1])
#         b_Sp1 = 0.0
#         cvec = hh.get_cons(r, w, b_s_vec, b_splus1_vec, nvec)
#     else:
#         rpath = r * np.ones(S)
#         wpath = w * np.ones(S)
#         c1_args = (0.0, beta, sigma, l_tilde, b_ellip, upsilon,
#                    chi_n_vec, rpath, wpath, diff)
#         c1_options = {'maxiter': 500}
#         results_c1 = \
#             opt.root(hh.c1_bSp1err, c1_init, args=(c1_args),
#                      method='lm', tol=1e-14, options=(c1_options))
#         c1 = results_c1.x
#         cnb_args = (0.0, beta, sigma, l_tilde, b_ellip, upsilon,
#                     chi_n_vec, diff)
#         cvec, nvec, b_s_vec, b_Sp1 = hh.get_cnb_vecs(c1, rpath, wpath,
#                                                   cnb_args)
#         b_splus1_vec = np.append(b_s_vec[1:],b_Sp1)

#     K, K_cnstr = aggr.get_K(b_s_vec)
#     L = aggr.get_L(nvec)
#     #K = np.maximum(0.0001, K)
#     L = np.maximum(0.0001, L)
#     r_params = (A, alpha, delta)
#     r_new = firms.get_r(r_params, K, L)
#     w_params = (A, alpha)
#     w_new = firms.get_w(w_params, K, L)

#     return K, L, cvec, nvec, b_s_vec, b_splus1_vec, b_Sp1, r_new, w_new

# def rw_errs(rwvec, *args):
#     '''
#     --------------------------------------------------------------------
#     Given values for r and w, solve for equilibrium errors from the two
#     first order conditions of the firm
#     --------------------------------------------------------------------
#     INPUTS:
#     rwvec = (2,) vector, (r, w)
#     args  = length 14 tuple, (c1_init, S, beta, sigma, l_tilde, b_ellip,
#             upsilon, chi_n_vec, A, alpha, delta, diff, hh_fsolve,
#             SS_tol)

#     OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
#         hh.bn_solve()
#         hh.c1_bSp1err()
#         hh.get_cnb_vecs()
#         aggr.get_K()
#         aggr.get_L()
#         firms.get_r()
#         firms.get_w()

#     OBJECTS CREATED WITHIN FUNCTION:
#     r          = scalar > 0, guess at steady-state interest rate
#     w          = scalar > 0, guess at steady-state wage
#     c1_init    = scalar > 0, initial guess for c1
#     S          = integer >= 3, number of periods in individual lifetime
#     beta       = scalar in (0,1), discount factor
#     sigma      = scalar >= 1, coefficient of relative risk aversion
#     l_tilde    = scalar > 0, per-period time endowment for every agent
#     b_ellip    = scalar > 0, fitted value of b for elliptical disutility
#                  of labor
#     upsilon    = scalar > 1, fitted value of upsilon for elliptical
#                  disutility of labor
#     chi_n_vec  = (S,) vector, values for chi^n_s
#     A          = scalar > 0, total factor productivity parameter in
#                  firms' production function
#     alpha      = scalar in (0,1), capital share of income
#     delta      = scalar in [0,1], model-period depreciation rate of
#                  capital
#     diff       = boolean, =True if simple difference Euler errors,
#                  otherwise percent deviation Euler errors
#     hh_fsolve  = boolean, =True if solve inner-loop household problem by
#                  choosing c_1 to set final period savings b_{S+1}=0.
#                  Otherwise, solve the household problem as multivariate
#                  root finder with 2S-1 unknowns and equations
#     SS_tol     = scalar > 0, tolerance level for steady-state fsolve
#     rpath      = (S,) vector, lifetime path of interest rates
#     wpath      = (S,) vector, lifetime path of wages
#     c1_args    = length 10 tuple, args to pass into c1_bSp1err()
#     c1_options = length 1 dict, options for c1_bSp1err()
#     results_c1 = results object, results from c1_bSp1err()
#     c1         = scalar > 0, optimal initial period consumption given r
#                  and w
#     cnb_args   = length 8 tuple, args to pass into get_cnb_vecs()
#     cvec       = (S,) vector, lifetime consumption (c1, c2, ...cS)
#     nvec       = (S,) vector, lifetime labor supply (n1, n2, ...nS)
#     bvec       = (S,) vector, lifetime savings (b1, b2, ...bS) with b1=0
#     b_Sp1      = scalar, final period savings, should be close to zero
#     K          = scalar > 0, aggregate capital stock
#     K_cnstr    = boolean, =True if K < 0
#     L          = scalar > 0, aggregate labor
#     r_params   = length 3 tuple, (A, alpha, delta)
#     EulErr_K   = scalar, Euler error from firm FOC for K
#     w_params   = length 2 tuple, (A, alpha)
#     EulErr_L   = scalar, Euler error from firm FOC for L
#     rw_errors  = (2,) vector, ([EulErr_K, EulErr_L])

#     FILES CREATED BY THIS FUNCTION: None

#     RETURNS: rw_errors
#     --------------------------------------------------------------------
#     '''
#     r, w = rwvec
#     (c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon,
#         chi_n_vec, A, alpha, delta, diff, hh_fsolve, SS_tol) = args

#     if (r + delta < 0) or (w <= 0):
#         rw_errors = np.array([1e14, 1e14])
#     else:
#         inner_loop_params = (c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon,
#             chi_n_vec, A, alpha, delta, diff, hh_fsolve, SS_tol)
#         K, L, cvec, nvec, b_s_vec, b_splus1_vec, b_Sp1, r_new, w_new\
#                = inner_loop(r, w, inner_loop_params)
#         EulErr_K = r - r_new
#         EulErr_L = w - w_new
#         rw_errors = np.array([EulErr_K, EulErr_L])

#         print('rw_errors: ', rw_errors)
#         print('rw: ', r, w)

#     return rw_errors


# def KL_errs(KLvec, *args):
#     '''
#     --------------------------------------------------------------------
#     Given values for w and r, solve for equilibrium errors from the two
#     first order conditions of the firm
#     --------------------------------------------------------------------
#     INPUTS:
#     rwvec = (2,) vector, (r, w)
#     args  = length 12 tuple, (c1_init, S, beta, sigma, l_tilde, b_ellip,
#             upsilon, chi_n_vec, A, alpha, delta, diff)

#     OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
#         hh.bn_solve()
#         hh.c1_bSp1err()
#         hh.get_cnb_vecs()
#         aggr.get_K()
#         aggr.get_L()
#         firms.get_r()
#         firms.get_w()

#     OBJECTS CREATED WITHIN FUNCTION:
#     r          = scalar > 0, guess at steady-state interest rate
#     w          = scalar > 0, guess at steady-state wage
#     c1_init    = scalar > 0, initial guess for c1
#     S          = integer >= 3, number of periods in individual lifetime
#     beta       = scalar in (0,1), discount factor
#     sigma      = scalar >= 1, coefficient of relative risk aversion
#     l_tilde    = scalar > 0, per-period time endowment for every agent
#     b_ellip    = scalar > 0, fitted value of b for elliptical disutility
#                  of labor
#     upsilon    = scalar > 1, fitted value of upsilon for elliptical
#                  disutility of labor
#     chi_n_vec  = (S,) vector, values for chi^n_s
#     A          = scalar > 0, total factor productivity parameter in
#                  firms' production function
#     alpha      = scalar in (0,1), capital share of income
#     delta      = scalar in [0,1], model-period depreciation rate of
#                  capital
#     diff       = boolean, =True if simple difference Euler errors,
#                  otherwise percent deviation Euler errors
#     rpath      = (S,) vector, lifetime path of interest rates
#     wpath      = (S,) vector, lifetime path of wages
#     c1_args    = length 10 tuple, args to pass into c1_bSp1err()
#     c1_options = length 1 dict, options for c1_bSp1err()
#     results_c1 = results object, results from c1_bSp1err()
#     c1         = scalar > 0, optimal initial period consumption given r
#                  and w
#     cnb_args   = length 8 tuple, args to pass into get_cnb_vecs()
#     cvec       = (S,) vector, lifetime consumption (c1, c2, ...cS)
#     nvec       = (S,) vector, lifetime labor supply (n1, n2, ...nS)
#     bvec       = (S,) vector, lifetime savings (b1, b2, ...bS) with b1=0
#     b_Sp1      = scalar, final period savings, should be close to zero
#     K          = scalar > 0, aggregate capital stock
#     K_cnstr    = boolean, =True if K < 0
#     L          = scalar > 0, aggregate labor
#     r_params   = length 3 tuple, (A, alpha, delta)
#     EulErr_K   = scalar, Euler error from firm FOC for K
#     w_params   = length 2 tuple, (A, alpha)
#     EulErr_L   = scalar, Euler error from firm FOC for L
#     rw_errors  = (2,) vector, ([EulErr_K, EulErr_L])

#     FILES CREATED BY THIS FUNCTION: None

#     RETURNS: rw_errors
#     --------------------------------------------------------------------
#     '''
#     K, L = KLvec
#     (c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon,
#         chi_n_vec, A, alpha, delta, diff, hh_fsolve, SS_tol) = args

#     if (K <= 0) or (L <= 0):
#         KL_errors = np.array([1e14, 1e14])
#     else:
#         r_params = (A, alpha, delta)
#         w_params = (A, alpha)
#         r = firms.get_r(r_params, K, L)
#         w = firms.get_w(w_params, K, L)

#         inner_loop_params = (c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon,
#             chi_n_vec, A, alpha, delta, diff, hh_fsolve, SS_tol)
#         K_new, L_new, cvec, nvec, b_s_vec, b_splus1_vec, b_Sp1, r_new, w_new\
#                = inner_loop(r, w, inner_loop_params)

#         KL_errors = np.array([K - K_new, L - L_new])
#         print('KL_errors: ', KL_errors)

#     return KL_errors


def SS_EulErrs(bvec, *args):
    '''
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    INPUTS:
    bvec = (S-1,) vector, lifetime savings
    args = length 7 tuple, (nvec, beta, sigma, A, alpha, delta, EulDiff)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        aggr.get_K()
        aggr.get_L()
        firms.get_r()
        firms.get_w()
        hh.get_cons()
        hh.get_b_errors()

    OBJECTS CREATED WITHIN FUNCTION:
    nvec     = (S,) vector, exogenous lifetime labor supply n_s
    beta     = scalar in (0,1), discount factor for each model per
    sigma    = scalar > 0, coefficient of relative risk aversion
    A        = scalar > 0, total factor productivity parameter in
               firms' production function
    alpha    = scalar in (0,1), capital share of income
    delta    = scalar in [0,1], model-period depreciation rate of
               capital
    EulDiff  = boolean, =True if want difference version of Euler errors
               beta*(1+r)*u'(c2) - u'(c1), =False if want ratio version
               version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    K        = scalar > 0, aggregate capital stock
    K_cstr   = boolean, =True if K < epsilon
    L        = scalar > 0, exogenous aggregate labor
    r_params = length 3 tuple, (A, alpha, delta)
    r        = scalar > 0, interest rate
    w_params = length 2 tuple, (A, alpha)
    w        = scalar > 0, wage
    c_args   = length 3 tuple, (nvec, r, w)
    cvec     = (S,) vector, household consumption c_s
    b_args   = length 4 tuple, (beta, sigma, r, EulDiff)
    errors   = (S-1,) vector, savings Euler errors given bvec

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: errors
    --------------------------------------------------------------------
    '''
    nvec, beta, sigma, A, alpha, delta, EulDiff = args
    K, K_cstr = aggr.get_K(bvec)
    L = aggr.get_L(nvec)
    r_params = (A, alpha, delta)
    r = firms.get_r(K, L, r_params)
    w_params = (A, alpha)
    w = firms.get_w(K, L, w_params)
    c_args = (nvec, r, w)
    cvec = hh.get_cons(bvec, 0.0, c_args)
    b_args = (beta, sigma, r, EulDiff)
    errors = hh.get_b_errors(cvec, b_args)

    return errors


def get_SS(bss_guess, args, graphs=False):
    '''
    --------------------------------------------------------------------
    Solve for the steady-state solution of the S-period-lived agent OG
    model with exogenous labor supply using one root finder in bvec
    --------------------------------------------------------------------
    INPUTS:
    bss_guess = (S-1,) vector, initial guess for b_ss
    args      = length 8 tuple,
                (nvec, beta, sigma, A, alpha, delta, SS_tol, SS_EulDiff)
    graphs    = boolean, =True if output steady-state graphs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        SS_EulErrs()
        aggr.get_K()
        aggr.get_L()
        aggr.get_Y()
        aggr.get_C()
        firms.get_r()
        firms.get_w()
        hh.get_cons()
        utils.print_time()
        get_ss_graphs()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time = scalar > 0, clock time at beginning of program
    nvec       = (S,) vector, exogenous lifetime labor supply n_s
    beta       = scalar in (0,1), discount factor for each model per
    sigma      = scalar > 0, coefficient of relative risk aversion
    A          = scalar > 0, total factor productivity parameter in
                 firms' production function
    alpha      = scalar in (0,1), capital share of income
    delta      = scalar in [0,1], model-period depreciation rate of
                 capital
    SS_tol     = scalar > 0, tolerance level for steady-state fsolve
    SS_EulDiff = Boolean, =True if want difference version of Euler
                 errors beta*(1+r)*u'(c2) - u'(c1), =False if want ratio
                 version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    b_args     = length 7 tuple, args passed to opt.root(SS_EulErrs,...)
    results_b  = results object, output from opt.root(SS_EulErrs,...)
    b_ss       = (S-1,) vector, steady-state savings b_{s+1}
    K_ss       = scalar > 0, steady-state aggregate capital stock
    Kss_cstr   = boolean, =True if K_ss < epsilon
    L          = scalar > 0, exogenous aggregate labor
    r_params   = length 3 tuple, (A, alpha, delta)
    r_ss       = scalar > 0, steady-state interest rate
    w_params   = length 2 tuple, (A, alpha)
    w_ss       = scalar > 0, steady-state wage
    c_args     = length 3 tuple, (nvec, r_ss, w_ss)
    c_ss       = (S,) vector, steady-state individual consumption c_s
    Y_params   = length 2 tuple, (A, alpha)
    Y_ss       = scalar > 0, steady-state aggregate output (GDP)
    C_ss       = scalar > 0, steady-state aggregate consumption
    b_err_ss   = (S-1,) vector, Euler errors associated with b_ss
    RCerr_ss   = scalar, steady-state resource constraint error
    ss_time    = scalar > 0, time elapsed during SS computation
                 (in seconds)
    ss_output  = length 10 dict, steady-state objects {b_ss, c_ss, w_ss,
                 r_ss, K_ss, Y_ss, C_ss, b_err_ss, RCerr_ss, ss_time}

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: ss_output
    --------------------------------------------------------------------
    '''
    start_time = time.clock()
    nvec, beta, sigma, A, alpha, delta, SS_tol, SS_EulDiff = args
    b_args = (nvec, beta, sigma, A, alpha, delta, SS_EulDiff)
    results_b = opt.root(SS_EulErrs, bss_guess, args=(b_args))
    b_ss = results_b.x
    K_ss, Kss_cstr = aggr.get_K(b_ss)
    L = aggr.get_L(nvec)
    r_params = (A, alpha, delta)
    r_ss = firms.get_r(K_ss, L, r_params)
    w_params = (A, alpha)
    w_ss = firms.get_w(K_ss, L, w_params)
    c_args = (nvec, r_ss, w_ss)
    c_ss = hh.get_cons(b_ss, 0.0, c_args)
    Y_params = (A, alpha)
    Y_ss = aggr.get_Y(K_ss, L, Y_params)
    C_ss = aggr.get_C(c_ss)
    b_err_ss = results_b.fun
    RCerr_ss = Y_ss - C_ss - delta * K_ss

    ss_time = time.clock() - start_time

    ss_output = {'b_ss': b_ss, 'c_ss': c_ss, 'w_ss': w_ss, 'r_ss': r_ss,
                 'K_ss': K_ss, 'Y_ss': Y_ss, 'C_ss': C_ss,
                 'b_err_ss': b_err_ss, 'RCerr_ss': RCerr_ss,
                 'ss_time': ss_time}
    print('b_ss is: ', b_ss)
    print('K_ss=', K_ss, ', r_ss=', r_ss, ', w_ss=', w_ss)
    print('Max. abs. savings Euler error is: ',
          np.absolute(b_err_ss).max())
    print('Max. abs. resource constraint error is: ',
          np.absolute(RCerr_ss).max())

    # Print SS computation time
    utils.print_time(ss_time, 'SS')

    if graphs:
        get_ss_graphs(c_ss, b_ss)

    return ss_output


# def get_SS_root(init_vals, args, graphs=False):
#     '''
#     --------------------------------------------------------------------
#     Solve for the steady-state solution of the S-period-lived agent OG
#     model with endogenous labor supply using the root finder in the
#     outer loop
#     --------------------------------------------------------------------
#     INPUTS:
#     init_vals = length 5 tuple,
#                 (Kss_init, Lss_init, rss_init, wss_init, c1_init)
#     args      = length 14 tuple, (S, beta, sigma, l_tilde, b_ellip,
#                 upsilon, chi_n_vec, A, alpha, delta, SS_tol, EulDiff,
#                 hh_fsolve, KL_outer)
#     graphs    = boolean, =True if output steady-state graphs

#     OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
#         rw_errs()
#         hh.bn_solve()
#         hh.c1_bSp1err()
#         hh.get_cnb_vecs()
#         aggr.get_K()
#         aggr.get_L()
#         aggr.get_Y()
#         aggr.get_C()
#         hh.get_cons()
#         hh.get_n_errors()
#         hh.get_b_errors()
#         utils.print_time()

#     OBJECTS CREATED WITHIN FUNCTION:
#     start_time   = scalar > 0, clock time at beginning of program
#     Kss_init     = scalar > 0, initial guess for steady-state aggregate
#                    capital stock
#     Lss_init     = scalar > 0, initial guess for steady-state aggregate
#                    labor
#     rss_init     = scalar > 0, initial guess for steady-state interest
#                    rate
#     wss_init     = scalar > 0, initial guess for steady-state wage
#     c1_init      = scalar > 0, initial guess for first period consumpt'n
#     S            = integer in [3, 80], number of periods an individual
#                    lives
#     beta         = scalar in (0,1), discount factor for each model per
#     sigma        = scalar > 0, coefficient of relative risk aversion
#     l_tilde      = scalar > 0, time endowment for each agent each period
#     b_ellip      = scalar > 0, fitted value of b for elliptical
#                    disutility of labor
#     upsilon      = scalar > 1, fitted value of upsilon for elliptical
#                    disutility of labor
#     chi_n_vec    = (S,) vector, values for chi^n_s
#     A            = scalar > 0, total factor productivity parameter in
#                    firms' production function
#     alpha        = scalar in (0,1), capital share of income
#     delta        = scalar in [0,1], model-period depreciation rate of
#                    capital
#     SS_tol       = scalar > 0, tolerance level for steady-state fsolve
#     EulDiff      = Boolean, =True if want difference version of Euler
#                    errors beta*(1+r)*u'(c2) - u'(c1), =False if want
#                    ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
#     hh_fsolve    = boolean, =True if solve inner-loop household problem
#                    by choosing c_1 to set final period savings b_{S+1}=0
#     KL_outer     = boolean, =True if guess K and L in outer loop.
#                    Otherwise, guess r and w in outer loop
#     rw_init      = (2,) vector, initial guesses for r_ss and w_ss
#     rw_args      = length 12 tuple, args to pass into rw_errs()
#     rw_options   = length 1 dict, options to pass into
#                    opt.root(rw_errs,...)
#     results_rw   = results object, root finder results from
#                    opt.root(rw_errs,...)
#     r_ss         = scalar > 0, steady-state interest rate
#     w_ss         = scalar > 0, steady-state wage
#     rpath        = (S,) vector, lifetime path of interest rates
#     wpath        = (S,) vector, lifetime path of wages
#     c1_args      = length 10 tuple, args to pass into c1_bSp1err()
#     c1_options   = length 1 dict, options to pass into
#                    opt.root(c1_bSp1err,...)
#     results_c1   = results object, root finder results from
#                    opt.root(c1_bSp1err,...)
#     c1_ss        = scalar > 0, steady-state consumption in first period
#     cnb_args     = length 8 tuple, args to pass into get_cnb_vecs()
#     c_ss         = (S,) vector, steady-state lifetime consumption
#     n_ss         = (S,) vector, steady-state lifetime labor supply
#     b_ss         = (S,) vector, steady-state lifetime savings
#                    (b1_ss, b2_ss, ...bS_ss) where b1_ss=0
#     b_Sp1_ss     = scalar, steady-state savings for period after last
#                    period of life. b_Sp1_ss approx. 0 in equilibrium
#     K_ss         = scalar > 0, steady-state aggregate capital stock
#     K_cnstr      = boolean, =True if K_ss <= 0
#     L_ss         = scalar > 0, steady-state aggregate labor
#     Y_params     = length 2 tuple, (A, alpha)
#     Y_ss         = scalar > 0, steady-state aggregate output (GDP)
#     C_ss         = scalar > 0, steady-state aggregate consumption
#     n_err_params = length 5 tuple, args to pass into get_n_errors()
#     n_err_ss     = (S,) vector, lifetime labor supply Euler errors
#     b_err_params = length 2 tuple, args to pass into get_b_errors()
#     b_err_ss     = (S) vector, lifetime savings Euler errors
#     RCerr_ss     = scalar, resource constraint error
#     ss_time      = scalar, seconds elapsed to run steady-state comput'n
#     ss_output    = length 14 dict, steady-state objects {n_ss, b_ss,
#                    c_ss, b_Sp1_ss, w_ss, r_ss, K_ss, L_ss, Y_ss, C_ss,
#                    n_err_ss, b_err_ss, RCerr_ss, ss_time}

#     FILES CREATED BY THIS FUNCTION:
#         None

#     RETURNS: ss_output
#     --------------------------------------------------------------------
#     '''
#     start_time = time.clock()
#     Kss_init, Lss_init, rss_init, wss_init, c1_init = init_vals
#     (S, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec, A, alpha,
#         delta, SS_tol, EulDiff, hh_fsolve, KL_outer) = args
#     if KL_outer:
#         KL_init = np.array([Kss_init, Lss_init])
#         KL_args = (c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon,
#                    chi_n_vec, A, alpha, delta, EulDiff, hh_fsolve,
#                    SS_tol)
#         KL_options = {'maxiter': 500}
#         results_KL = \
#             opt.root(KL_errs, KL_init, args=(KL_args), method='lm',
#                      tol=SS_tol, options=(KL_options))
#         K_ss, L_ss = results_KL.x
#         r_params = (A, alpha, delta)
#         w_params = (A, alpha)
#         r_ss = firms.get_r(r_params, K_ss, L_ss)
#         w_ss = firms.get_w(w_params, K_ss, L_ss)
#     else:
#         rw_init = np.array([rss_init, wss_init])
#         rw_args = (c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon,
#                    chi_n_vec, A, alpha, delta, EulDiff, hh_fsolve,
#                    SS_tol)
#         rw_options = {'maxiter': 500}
#         results_rw = \
#             opt.root(rw_errs, rw_init, args=(rw_args), method='lm',
#                      tol=SS_tol, options=(rw_options))
#         r_ss, w_ss = results_rw.x

#     inner_loop_params = (c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon,
#         chi_n_vec, A, alpha, delta, EulDiff, hh_fsolve, SS_tol)
#     K_ss, L_ss, c_ss, n_ss, b_s_ss, b_splus1_ss, b_Sp1_ss, r_ss, w_ss\
#            = inner_loop(r_ss, w_ss, inner_loop_params)

#     Y_params = (A, alpha)
#     Y_ss = aggr.get_Y(Y_params, K_ss, L_ss)
#     C_ss = aggr.get_C(c_ss)
#     n_err_args = (w_ss, c_ss, sigma, l_tilde, chi_n_vec, b_ellip, upsilon, EulDiff)
#     n_err_ss = hh.get_n_errors(n_ss, n_err_args)
#     b_err_params = (beta, sigma)
#     b_err_ss = hh.get_b_errors(b_err_params, r_ss, c_ss, EulDiff)
#     RCerr_ss = Y_ss - C_ss - delta * K_ss

#     ss_time = time.clock() - start_time

#     ss_output = {
#         'n_ss': n_ss, 'b_s_ss': b_s_ss, 'b_splus1_ss': b_splus1_ss,
#         'c_ss': c_ss, 'b_Sp1_ss': b_Sp1_ss,
#         'w_ss': w_ss, 'r_ss': r_ss, 'K_ss': K_ss, 'L_ss': L_ss,
#         'Y_ss': Y_ss, 'C_ss': C_ss, 'n_err_ss': n_err_ss,
#         'b_err_ss': b_err_ss, 'RCerr_ss': RCerr_ss, 'ss_time': ss_time}
#     print('n_ss is: ', n_ss)
#     print('b_splus1_ss is: ', b_splus1_ss)
#     print('K_ss=', K_ss, ', L_ss=', L_ss)
#     print('r_ss=', r_ss, ', w_ss=', w_ss)
#     print('Maximum abs. labor supply Euler error is: ',
#           np.absolute(n_err_ss).max())
#     print('Maximum abs. savings Euler error is: ',
#           np.absolute(b_err_ss).max())
#     print('Resource constraint error is: ', RCerr_ss)
#     print('Steady-state residual savings b_Sp1 is: ', b_Sp1_ss)

#     # Print SS computation time
#     utils.print_time(ss_time, 'SS')

#     if graphs:
#         create_graphs(c_ss, b_ss)

#     return ss_output


# def get_SS_bsct(init_vals, args, graphs=False):
#     '''
#     --------------------------------------------------------------------
#     Solve for the steady-state solution of the S-period-lived agent OG
#     model with endogenous labor supply using the bisection method for
#     the outer loop
#     --------------------------------------------------------------------
#     INPUTS:
#     init_vals = length 5 tuple,
#                 (Kss_init, Lss_init, rss_init, wss_init,c1_init)
#     args      = length 14 tuple, (S, beta, sigma, l_tilde, b_ellip,
#                 upsilon, chi_n_vec, A, alpha, delta, SS_tol, EulDiff,
#                 hh_fsolve, KL_outer)
#     graphs    = boolean, =True if output steady-state graphs

#     OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
#         firms.get_r()
#         firms.get_w()
#         hh.bn_solve()
#         hh.c1_bSp1err()
#         hh.get_cnb_vecs()
#         aggr.get_K()
#         aggr.get_L()
#         aggr.get_Y()
#         aggr.get_C()
#         hh.get_cons()
#         hh.get_n_errors()
#         hh.get_b_errors()
#         utils.print_time()

#     OBJECTS CREATED WITHIN FUNCTION:
#     start_time   = scalar > 0, clock time at beginning of program
#     Kss_init     = scalar > 0, initial guess for steady-state aggregate
#                    capital stock
#     Lss_init     = scalar > 0, initial guess for steady-state aggregate
#                    labor
#     rss_init     = scalar > 0, initial guess for steady-state interest
#                    rate
#     wss_init     = scalar > 0, initial guess for steady-state wage
#     c1_init      = scalar > 0, initial guess for first period consumpt'n
#     S            = integer in [3, 80], number of periods an individual
#                    lives
#     beta         = scalar in (0,1), discount factor for each model per
#     sigma        = scalar > 0, coefficient of relative risk aversion
#     l_tilde      = scalar > 0, time endowment for each agent each period
#     b_ellip      = scalar > 0, fitted value of b for elliptical
#                    disutility of labor
#     upsilon      = scalar > 1, fitted value of upsilon for elliptical
#                    disutility of labor
#     chi_n_vec    = (S,) vector, values for chi^n_s
#     A            = scalar > 0, total factor productivity parameter in
#                    firms' production function
#     alpha        = scalar in (0,1), capital share of income
#     delta        = scalar in [0,1], model-period depreciation rate of
#                    capital
#     SS_tol       = scalar > 0, tolerance level for steady-state fsolve
#     EulDiff      = Boolean, =True if want difference version of Euler
#                    errors beta*(1+r)*u'(c2) - u'(c1), =False if want
#                    ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
#     hh_fsolve    = boolean, =True if solve inner-loop household problem
#                    by choosing c_1 to set final period savings b_{S+1}=0
#     KL_outer     = boolean, =True if guess K and L in outer loop.
#                    Otherwise, guess r and w in outer loop
#     maxiter_SS   = integer >= 1, maximum number of iterations in outer
#                    loop bisection method
#     iter_SS      = integer >= 0, index of iteration number
#     mindist_SS   = scalar > 0, minimum distance tolerance for
#                    convergence
#     dist_SS      = scalar > 0, distance metric for current iteration
#     xi_SS        = scalar in (0,1], updating parameter
#     KL_init      = (2,) vector, (K_init, L_init)
#     c1_options   = length 1 dict, options to pass into
#                    opt.root(c1_bSp1err,...)
#     cnb_args     = length 8 tuple, args to pass into get_cnb_vecs()
#     r_params     = length 3 tuple, args to pass into get_r()
#     w_params     = length 2 tuple, args to pass into get_w()
#     K_init       = scalar, initial value of aggregate capital stock
#     L_init       = scalar, initial value of aggregate labor
#     r_init       = scalar, initial value for interest rate
#     w_init       = scalar, initial value for wage
#     rpath        = (S,) vector, lifetime path of interest rates
#     wpath        = (S,) vector, lifetime path of wages
#     c1_args      = length 10 tuple, args to pass into c1_bSp1err()
#     results_c1   = results object, root finder results from
#                    opt.root(c1_bSp1err,...)
#     c1_new       = scalar, updated value of optimal c1 given r_init and
#                    w_init
#     cvec_new     = (S,) vector, updated values for lifetime consumption
#     nvec_new     = (S,) vector, updated values for lifetime labor supply
#     bvec_new     = (S,) vector, updated values for lifetime savings
#                    (b1, b2,...bS)
#     b_Sp1_new    = scalar, updated value for savings in last period,
#                    should be arbitrarily close to zero
#     K_new        = scalar, updated K given bvec_new
#     K_cnstr      = boolean, =True if K_new <= 0
#     L_new        = scalar, updated L given nvec_new
#     KL_new       = (2,) vector, updated K and L given bvec_new, nvec_new
#     K_ss         = scalar > 0, steady-state aggregate capital stock
#     L_ss         = scalar > 0, steady-state aggregate labor
#     r_ss         = scalar > 0, steady-state interest rate
#     w_ss         = scalar > 0, steady-state wage
#     c1_ss        = scalar > 0, steady-state consumption in first period
#     c_ss         = (S,) vector, steady-state lifetime consumption
#     n_ss         = (S,) vector, steady-state lifetime labor supply
#     b_s_ss       = (S,) vector, steady-state wealth enter period with
#     b_splus1_ss  = (S,) vector, steady-state lifetime savings
#                    (b1_ss, b2_ss, ...bS_ss) where b1_ss=0
#     b_Sp1_ss     = scalar, steady-state savings for period after last
#                    period of life. b_Sp1_ss approx. 0 in equilibrium
#     Y_params     = length 2 tuple, (A, alpha)
#     Y_ss         = scalar > 0, steady-state aggregate output (GDP)
#     C_ss         = scalar > 0, steady-state aggregate consumption
#     n_err_params = length 5 tuple, args to pass into get_n_errors()
#     n_err_ss     = (S,) vector, lifetime labor supply Euler errors
#     b_err_params = length 2 tuple, args to pass into get_b_errors()
#     b_err_ss     = (S-1) vector, lifetime savings Euler errors
#     RCerr_ss     = scalar, resource constraint error
#     ss_time      = scalar, seconds elapsed to run steady-state comput'n
#     ss_output    = length 14 dict, steady-state objects {n_ss, b_ss,
#                    c_ss, b_Sp1_ss, w_ss, r_ss, K_ss, L_ss, Y_ss, C_ss,
#                    n_err_ss, b_err_ss, RCerr_ss, ss_time}

#     FILES CREATED BY THIS FUNCTION:
#         SS_bc.png
#         SS_n.png

#     RETURNS: ss_output
#     --------------------------------------------------------------------
#     '''
#     start_time = time.clock()
#     Kss_init, Lss_init, rss_init, wss_init, c1_init = init_vals
#     (S, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec, A, alpha,
#         delta, SS_tol, EulDiff, hh_fsolve, KL_outer) = args
#     maxiter_SS = 200
#     iter_SS = 0
#     mindist_SS = 1e-10
#     dist_SS = 10
#     xi_SS = 0.15
#     KL_init = np.array([Kss_init, Lss_init])
#     rw_init = np.array([rss_init, wss_init])
#     c1_options = {'maxiter': 500}
#     cnb_args = (0.0, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec,
#                 EulDiff)
#     r_params = (A, alpha, delta)
#     w_params = (A, alpha)
#     while (iter_SS < maxiter_SS) and (dist_SS >= mindist_SS):
#         iter_SS += 1
#         if KL_outer:
#             K_init, L_init = KL_init
#             r_init = firms.get_r(r_params, K_init, L_init)
#             w_init = firms.get_w(w_params, K_init, L_init)

#             inner_loop_params = (c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon,
#                 chi_n_vec, A, alpha, delta, EulDiff, hh_fsolve, SS_tol)
#             K_new, L_new, cvec_new, nvec_new, b_s_vec_new, b_splus1_vec_new,\
#                     b_Sp1_vec_new, r_new, w_new \
#                     = inner_loop(r_init, w_init, inner_loop_params)

#             #K_new = np.maximum(0.0001, K_new)
#             #L_new = np.maximum(0.0001, L_new)
#             print('K and L: ', K_new, L_new)
#             KL_new = np.array([K_new, L_new])
#             dist_SS = ((KL_new - KL_init) ** 2).sum()
#             KL_init = xi_SS * KL_new + (1 - xi_SS) * KL_init
#         else:
#             r_init, w_init = rw_init

#             inner_loop_params = (c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon,
#                 chi_n_vec, A, alpha, delta, EulDiff, hh_fsolve, SS_tol)
#             K_new, L_new, cvec_new, nvec_new, b_s_vec_new, b_splus1_vec_new,\
#                     b_Sp1_vec_new, r_new, w_new \
#                     = inner_loop(r_init, w_init, inner_loop_params)

#             rw_new = np.array([r_new, w_new])
#             print('r init and new: ', r_init, r_new)
#             print('w init and new: ', w_init, w_new)
#             dist_SS = (((rw_new - rw_init)*1000) ** 2).sum()
#             rw_init = xi_SS * rw_new + (1 - xi_SS) * rw_init

#         print('SS Iteration=', iter_SS, ', SS Distance=', dist_SS)

#     inner_loop_params = (c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon,
#         chi_n_vec, A, alpha, delta, EulDiff, hh_fsolve, SS_tol)
#     K_ss, L_ss, c_ss, n_ss, b_s_ss, b_splus1_ss, b_Sp1_ss, r_ss, w_ss\
#            = inner_loop(r_new, w_new, inner_loop_params)

#     Y_params = (A, alpha)
#     Y_ss = aggr.get_Y(Y_params, K_ss, L_ss)
#     C_ss = aggr.get_C(c_ss)
#     n_err_args = (w_ss, c_ss, sigma, l_tilde, chi_n_vec, b_ellip, upsilon, EulDiff)
#     n_err_ss = hh.get_n_errors(n_ss, n_err_args)
#     b_err_params = (beta, sigma)
#     b_err_ss = hh.get_b_errors(b_err_params, r_ss, c_ss, EulDiff)
#     RCerr_ss = Y_ss - C_ss - delta * K_ss

#     ss_time = time.clock() - start_time

#     ss_output = {
#         'n_ss': n_ss, 'b_s_ss': b_s_ss, 'b_splus1_ss': b_splus1_ss,
#         'c_ss': c_ss, 'b_Sp1_ss': b_Sp1_ss,
#         'w_ss': w_ss, 'r_ss': r_ss, 'K_ss': K_ss, 'L_ss': L_ss,
#         'Y_ss': Y_ss, 'C_ss': C_ss, 'n_err_ss': n_err_ss,
#         'b_err_ss': b_err_ss, 'RCerr_ss': RCerr_ss, 'ss_time': ss_time}
#     print('n_ss is: ', n_ss)
#     print('b_splus1_ss is: ', b_splus1_ss)
#     print('K_ss=', K_ss, ', L_ss=', L_ss)
#     print('r_ss=', r_ss, ', w_ss=', w_ss)
#     print('Maximum abs. labor supply Euler error is: ',
#           np.absolute(n_err_ss).max())
#     print('Maximum abs. savings Euler error is: ',
#           np.absolute(b_err_ss).max())
#     print('Resource constraint error is: ', RCerr_ss)
#     print('Steady-state residual savings b_Sp1 is: ', b_Sp1_ss)

#     # Print SS computation time
#     utils.print_time(ss_time, 'SS')

#     if graphs:
#         create_graphs(c_ss, b_splus1_ss, n_ss, S)

#     return ss_output
