'''
------------------------------------------------------------------------
This module contains the functions used to solve the steady state of
the model with S-period lived agents and exogenous labor supply from
Chapter 7 of the OG textbook.

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
