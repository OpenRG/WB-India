'''
------------------------------------------------------------------------
This module contains the functions used to solve the steady state of
the model with S-period lived agents, endogenous labor supply, and
heterogeneous ability from Chapter 8 of the OG textbook.

This Python module imports the following module(s):
    households.py
    firms.py
    aggregates.py
    utilities.py

This Python module defines the following function(s):
    get_SS_bsct()
------------------------------------------------------------------------
'''
# Import packages
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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


def euler_sys(guesses, *args):
    '''
    --------------------------------------------------------------------
    Specify the system of Euler Equations characterizing the household
    problem.
    --------------------------------------------------------------------
    INPUTS:
    guesses = (2S-1,) vector, guess at labor supply and savings decisions
    args = length 14 tuple, (r, w, beta, sigma, l_tilde, chi_n_vec,
            b_ellip, upsilon, diff, S, SS_tol)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        hh.get_n_errors()
        hh.get_n_errors()
        hh.get_cons()

    OBJECTS CREATED WITHIN FUNCTION:
    r    = scalar > 0, guess at steady-state interest rate
    w    = scalar > 0, guess at steady-state wage
    beta       = scalar in (0,1), discount factor
    sigma      = scalar >= 1, coefficient of relative risk aversion
    l_tilde    = scalar > 0, per-period time endowment for every agent
    chi_n_vec  = (S,) vector, values for chi^n_s
    b_ellip    = scalar > 0, fitted value of b for elliptical disutility
                 of labor
    upsilon    = scalar > 1, fitted value of upsilon for elliptical
                 disutility of labor
    A          = scalar > 0, total factor productivity parameter in
                 firms' production function
    diff    = boolean, =True if simple difference Euler errors,
                 otherwise percent deviation Euler errors
    S          = integer >= 3, number of periods in individual lifetime
    SS_tol     = scalar > 0, tolerance level for steady-state fsolve
    nvec       = (S,) vector, lifetime labor supply (n1, n2, ...nS)
    bvec       = (S,) vector, lifetime savings (b1, b2, ...bS) with b1=0
    cvec       = (S,) vector, lifetime consumption (c1, c2, ...cS)
    b_sp1      = = (S,) vector, lifetime savings (b1, b2, ...bS) with bS=0
    n_errors   = (S,) vector, labor supply Euler errors
    b_errors   = (S-1,) vector, savings Euler errors

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: array of n_errors and b_errors
    --------------------------------------------------------------------
    '''
    (r, w, beta, sigma, emat, l_tilde, chi_n_vec, b_ellip, upsilon,
     diff, S, SS_tol) = args
    nvec = guesses[:S]
    bvec1 = guesses[S:]

    b_s = np.append(0.0, bvec1)
    b_sp1 = np.append(bvec1, 0.0)
    cvec = hh.get_cons(r, w, b_s, b_sp1, nvec, emat)
    n_args = (w, sigma, emat, l_tilde, chi_n_vec, b_ellip, upsilon, diff,
              cvec)
    n_errors = hh.get_n_errors(nvec, *n_args)
    b_args = (r, beta, sigma, diff)
    b_errors = hh.get_b_errors(cvec, *b_args)

    errors = np.append(n_errors, b_errors)

    return errors


def inner_loop(r, w, args):
    '''
    --------------------------------------------------------------------
    Given values for r and w, solve for the households' optimal decisions
    --------------------------------------------------------------------
    INPUTS:
    r    = scalar > 0, guess at steady-state interest rate
    w    = scalar > 0, guess at steady-state wage
    args = length 14 tuple, (nvec_init, bvec_init, S, beta, sigma,
            l_tilde, b_ellip, upsilon, chi_n_vec, A, alpha, delta,
            EulDiff, SS_tol)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        euler_sys()
        aggr.get_K()
        aggr.get_L()
        firms.get_r()
        firms.get_w()

    OBJECTS CREATED WITHIN FUNCTION:
    nvec_init  = (S,) vector, initial guesses at choice of labor supply
    bvec_init  = (S,) vector, initial guesses at choice of savings
    S          = integer >= 3, number of periods in individual lifetime
    beta       = scalar in (0,1), discount factor
    sigma      = scalar >= 1, coefficient of relative risk aversion
    l_tilde    = scalar > 0, per-period time endowment for every agent
    b_ellip    = scalar > 0, fitted value of b for elliptical disutility
                 of labor
    upsilon    = scalar > 1, fitted value of upsilon for elliptical
                 disutility of labor
    chi_n_vec  = (S,) vector, values for chi^n_s
    A          = scalar > 0, total factor productivity parameter in
                 firms' production function
    alpha      = scalar in (0,1), capital share of income
    delta      = scalar in [0,1], model-period depreciation rate of
                 capital
    EulDiff    = boolean, =True if simple difference Euler errors,
                 otherwise percent deviation Euler errors
    SS_tol     = scalar > 0, tolerance level for steady-state fsolve
    cvec       = (S,) vector, lifetime consumption (c1, c2, ...cS)
    nvec       = (S,) vector, lifetime labor supply (n1, n2, ...nS)
    bvec       = (S,) vector, lifetime savings (b1, b2, ...bS) with b1=0
    b_Sp1      = scalar, final period savings, should be close to zero
    n_errors   = (S,) vector, labor supply Euler errors
    b_errors   = (S-1,) vector, savings Euler errors
    K          = scalar > 0, aggregate capital stock
    K_cstr     = boolean, =True if K < epsilon
    L          = scalar > 0, aggregate labor
    L_cstr     = boolean, =True if L < epsilon
    r_params   = length 3 tuple, (A, alpha, delta)
    w_params   = length 2 tuple, (A, alpha)
    r_new      = scalar > 0, guess at steady-state interest rate
    w_new      = scalar > 0, guess at steady-state wage

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: K, L, cvec, nvec, bvec, b_Sp1, r_new, w_new, n_errors,
             b_errors
    --------------------------------------------------------------------
    '''
    (nmat_init, bmat_init, S, J, beta, sigma, emat, l_tilde, b_ellip,
     upsilon, chi_n_vec, A, alpha, delta, lambdas, EulDiff,
     SS_tol) = args

    nmat = np.zeros((S, J))
    bmat = np.zeros((S - 1, J))
    n_err_mat = np.zeros((S, J))
    b_err_mat = np.zeros((S - 1, J))
    for j in range(J):
        euler_args = (r, w, beta, sigma, emat[:, j], l_tilde, chi_n_vec,
                      b_ellip, upsilon, EulDiff, S, SS_tol)
        guesses = np.append(nmat_init[:, j], bmat_init[:, j])
        results_euler = opt.root(euler_sys, guesses, args=(euler_args),
                                 method='lm', tol=SS_tol)
        nmat[:, j] = results_euler.x[:S]
        bmat[:, j] = results_euler.x[S:]
        n_err_mat[:, j] = results_euler.fun[:S]
        b_err_mat[:, j] = results_euler.fun[S:]
    b_s_mat = np.append(np.zeros((1, J)), bmat, axis=0)
    b_sp1_mat = np.append(bmat, np.zeros((1, J)), axis=0)
    cmat = hh.get_cons(r, w, b_s_mat, b_sp1_mat, nmat, emat)
    b_Sp1_vec = np.zeros(J)
    K, K_cnstr = aggr.get_K(bmat, lambdas)
    L = aggr.get_L(nmat, emat, lambdas)
    r_params = (A, alpha, delta)
    r_new = firms.get_r(K, L, r_params)
    w_params = (A, alpha, delta)
    w_new = firms.get_w_from_r(r_new, w_params)

    return (K, L, cmat, nmat, bmat, b_Sp1_vec, r_new, w_new,
            n_err_mat, b_err_mat)


def get_SS_bsct(init_vals, args, graphs=False):
    '''
    --------------------------------------------------------------------
    Solve for the steady-state solution of the S-period-lived agent OG
    model with endogenous labor supply using the bisection method for
    the outer loop
    --------------------------------------------------------------------
    INPUTS:
    init_vals = length 3 tuple, (Kss_init, Lss_init, c1_init)
    args      = length 15 tuple, (J, S, lambdas, emat, beta, sigma,
                l_tilde, b_ellip, upsilon, chi_n_vec, A, alpha, delta,
                SS_tol, EulDiff)
    graphs    = boolean, =True if output steady-state graphs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        firms.get_r()
        firms.get_w()
        hh.bn_solve()
        hh.c1_bSp1err()
        hh.get_cnb_vecs()
        aggr.get_K()
        aggr.get_L()
        aggr.get_Y()
        aggr.get_C()
        hh.get_cons()
        hh.get_n_errors()
        hh.get_b_errors()
        utils.print_time()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time   = scalar > 0, clock time at beginning of program
    rss_init     = scalar > 0, initial guess for steady-state interest
                   rate
    c1_init      = scalar > 0, initial guess for first period consumpt'n
    S            = integer in [3, 80], number of periods an individual
                   lives
    beta         = scalar in (0,1), discount factor for each model per
    sigma        = scalar > 0, coefficient of relative risk aversion
    l_tilde      = scalar > 0, time endowment for each agent each period
    b_ellip      = scalar > 0, fitted value of b for elliptical
                   disutility of labor
    upsilon      = scalar > 1, fitted value of upsilon for elliptical
                   disutility of labor
    chi_n_vec    = (S,) vector, values for chi^n_s
    A            = scalar > 0, total factor productivity parameter in
                   firms' production function
    alpha        = scalar in (0,1), capital share of income
    delta        = scalar in [0,1], model-period depreciation rate of
                   capital
    SS_tol       = scalar > 0, tolerance level for steady-state fsolve
    EulDiff      = Boolean, =True if want difference version of Euler
                   errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                   ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    hh_fsolve    = boolean, =True if solve inner-loop household problem
                   by choosing c_1 to set final period savings b_{S+1}=0
    KL_outer     = boolean, =True if guess K and L in outer loop
                   Otherwise, guess r and w in outer loop
    maxiter_SS   = integer >= 1, maximum number of iterations in outer
                   loop bisection method
    iter_SS      = integer >= 0, index of iteration number
    mindist_SS   = scalar > 0, minimum distance tolerance for
                   convergence
    dist_SS      = scalar > 0, distance metric for current iteration
    xi_SS        = scalar in (0,1], updating parameter
    KL_init      = (2,) vector, (K_init, L_init)
    c1_options   = length 1 dict, options to pass into
                   opt.root(c1_bSp1err,...)
    cnb_args     = length 8 tuple, args to pass into get_cnb_vecs()
    r_params     = length 3 tuple, args to pass into get_r()
    w_params     = length 2 tuple, args to pass into get_w()
    K_init       = scalar, initial value of aggregate capital stock
    L_init       = scalar, initial value of aggregate labor
    r_init       = scalar, initial value for interest rate
    w_init       = scalar, initial value for wage
    rpath        = (S,) vector, lifetime path of interest rates
    wpath        = (S,) vector, lifetime path of wages
    c1_args      = length 10 tuple, args to pass into c1_bSp1err()
    results_c1   = results object, root finder results from
                   opt.root(c1_bSp1err,...)
    c1_new       = scalar, updated value of optimal c1 given r_init and
                   w_init
    cvec_new     = (S,) vector, updated values for lifetime consumption
    nvec_new     = (S,) vector, updated values for lifetime labor supply
    bvec_new     = (S,) vector, updated values for lifetime savings
                   (b1, b2,...bS)
    b_Sp1_new    = scalar, updated value for savings in last period,
                   should be arbitrarily close to zero
    K_new        = scalar, updated K given bvec_new
    K_cnstr      = boolean, =True if K_new <= 0
    L_new        = scalar, updated L given nvec_new
    KL_new       = (2,) vector, updated K and L given bvec_new, nvec_new
    K_ss         = scalar > 0, steady-state aggregate capital stock
    L_ss         = scalar > 0, steady-state aggregate labor
    r_ss         = scalar > 0, steady-state interest rate
    w_ss         = scalar > 0, steady-state wage
    c1_ss        = scalar > 0, steady-state consumption in first period
    c_ss         = (S,) vector, steady-state lifetime consumption
    n_ss         = (S,) vector, steady-state lifetime labor supply
    b_ss         = (S,) vector, steady-state lifetime savings
                   (b1_ss, b2_ss, ...bS_ss) where b1_ss=0
    b_Sp1_ss     = scalar, steady-state savings for period after last
                   period of life. b_Sp1_ss approx. 0 in equilibrium
    Y_params     = length 2 tuple, (A, alpha)
    Y_ss         = scalar > 0, steady-state aggregate output (GDP)
    C_ss         = scalar > 0, steady-state aggregate consumption
    n_err_params = length 5 tuple, args to pass into get_n_errors()
    n_err_ss     = (S,) vector, lifetime labor supply Euler errors
    b_err_params = length 2 tuple, args to pass into get_b_errors()
    b_err_ss     = (S-1) vector, lifetime savings Euler errors
    RCerr_ss     = scalar, resource constraint error
    ss_time      = scalar, seconds elapsed to run steady-state comput'n
    ss_output    = length 14 dict, steady-state objects {n_ss, b_ss,
                   c_ss, b_Sp1_ss, w_ss, r_ss, K_ss, L_ss, Y_ss, C_ss,
                   n_err_ss, b_err_ss, RCerr_ss, ss_time}

    FILES CREATED BY THIS FUNCTION:
        SS_bc.png
        SS_n.png

    RETURNS: ss_output
    --------------------------------------------------------------------
    '''
    start_time = time.clock()
    r_init, c1_init = init_vals
    (J, S, lambdas, emat, beta, sigma, l_tilde, b_ellip, upsilon,
        chi_n_vec, A, alpha, delta, SS_tol, EulDiff) = args
    maxiter_SS = 200
    iter_SS = 0
    mindist_SS = 1e-12
    dist_SS = 10
    xi_SS = 0.2
    c1_options = {'maxiter': 500}
    rw_params = (A, alpha, delta)
    nmat_init = np.ones((S, J)) * 0.4
    bmat_init = np.ones((S - 1, J)) * 0.03
    nmat = nmat_init
    bmat = bmat_init
    while (iter_SS < maxiter_SS) and (dist_SS >= mindist_SS):
        iter_SS += 1
        w_init = firms.get_w_from_r(r_init, rw_params)
        inner_args = (nmat, bmat, S, J, beta, sigma, emat, l_tilde,
                      b_ellip, upsilon, chi_n_vec, A, alpha, delta,
                      lambdas, EulDiff, SS_tol)
        (K_new, L_new, cmat, nmat, bmat, b_Sp1_vec, r_new, w_new,
         n_err_mat, b_err_mat) = inner_loop(r_init, w_init, inner_args)
        dist_SS = np.absolute(r_new - r_init).sum()
        r_init = xi_SS * r_new + (1 - xi_SS) * r_init
        print('SS Iteration=', iter_SS, ', SS Distance=',
              '%10.4e' % (dist_SS), ',r:', '%10.4e' % (r_new),
              ', Max Abs Errors=', '%10.4e' %
              (np.absolute(n_err_mat).max()),
              (np.absolute(n_err_mat).max()))

    K_ss = K_new
    L_ss = L_new
    r_ss = r_init
    w_ss = firms.get_w_from_r(r_ss, rw_params)
    c_ss = cmat
    n_ss = nmat
    b_ss = np.append(np.zeros((1, J)), bmat, axis=0)
    b_Sp1_ss = b_Sp1_vec
    Y_params = (A, alpha)
    Y_ss = aggr.get_Y(K_ss, L_ss, Y_params)
    C_ss = aggr.get_C(c_ss, lambdas)
    n_err_ss = n_err_mat
    b_err_ss = b_err_mat

    RCerr_ss = Y_ss - C_ss - delta * K_ss

    ss_time = time.clock() - start_time

    ss_output = {
        'n_ss': n_ss, 'b_ss': b_ss, 'c_ss': c_ss, 'b_Sp1_ss': b_Sp1_ss,
        'w_ss': w_ss, 'r_ss': r_ss, 'K_ss': K_ss, 'L_ss': L_ss,
        'Y_ss': Y_ss, 'C_ss': C_ss, 'n_err_ss': n_err_ss,
        'b_err_ss': b_err_ss, 'RCerr_ss': RCerr_ss, 'ss_time': ss_time}
    print('K_ss=', K_ss, ', L_ss=', L_ss)
    print('r_ss=', r_ss, ', w_ss=', w_ss)
    print('Maximum abs. labor supply Euler error is: ',
          np.absolute(n_err_ss).max())
    print('Maximum abs. savings Euler error is: ',
          np.absolute(b_err_ss).max())
    print('Resource constraint error is: ', RCerr_ss)
    print('Max. absolute SS residual savings b_Sp1_j is: ',
          np.absolute(b_Sp1_ss).max())

    # Print SS computation time
    utils.print_time(ss_time, 'SS')

    if graphs:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        sgrid       = (S,) vector, ages from 1 to S
        lamcumsum   = (J,) vector, cumulative sum of lambdas vector
        jmidgrid    = (J,) vector, midpoints of ability percentile bins
        smat        = (J, S) matrix, sgrid copied down J rows
        jmat        = (J, S) matrix, jmidgrid copied across S columns
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = 'images'
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot 3D steady-state consumption distribution
        sgrid = np.arange(1, S + 1)
        lamcumsum = lambdas.cumsum()
        jmidgrid = 0.5 * lamcumsum + 0.5 * (lamcumsum - lambdas)
        smat, jmat = np.meshgrid(sgrid, jmidgrid)
        cmap_c = cm.get_cmap('summer')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'ability-$j$')
        ax.set_zlabel(r'indiv. consumption $c_{j,s}$')
        ax.plot_surface(smat, jmat, c_ss.T, rstride=1,
                        cstride=6, cmap=cmap_c)
        output_path = os.path.join(output_dir, 'c_ss_3D')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot 2D steady-state consumption distribution
        minorLocator = MultipleLocator(1)
        fig, ax = plt.subplots()
        linestyles = np.array(["-", "--", "-.", ":"])
        markers = np.array(["x", "v", "o", "d", ">", "|"])
        pct_lb = 0
        for j in range(J):
            this_label = (str(int(np.rint(pct_lb))) + " - " +
                          str(int(np.rint(pct_lb + 100 * lambdas[j]))) +
                          "%")
            pct_lb += 100 * lambdas[j]
            if j <= 3:
                ax.plot(sgrid, c_ss[:, j], label=this_label,
                        linestyle=linestyles[j], color='black')
            elif j > 3:
                ax.plot(sgrid, c_ss[:, j], label=this_label,
                        marker=markers[j - 4], color='black')
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'indiv. consumption $c_{j,s}$')
        output_path = os.path.join(output_dir, 'c_ss_2D')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot 3D steady-state labor supply distribution
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'ability-$j$')
        ax.set_zlabel(r'labor supply $n_{j,s}$')
        ax.plot_surface(smat, jmat, n_ss.T, rstride=1,
                        cstride=6, cmap=cmap_c)
        output_path = os.path.join(output_dir, 'n_ss_3D')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot 2D steady-state labor supply distribution
        minorLocator = MultipleLocator(1)
        fig, ax = plt.subplots()
        linestyles = np.array(["-", "--", "-.", ":"])
        markers = np.array(["x", "v", "o", "d", ">", "|"])
        pct_lb = 0
        for j in range(J):
            this_label = (str(int(np.rint(pct_lb))) + " - " +
                          str(int(np.rint(pct_lb + 100 * lambdas[j]))) +
                          "%")
            pct_lb += 100 * lambdas[j]
            if j <= 3:
                ax.plot(sgrid, n_ss[:, j], label=this_label,
                        linestyle=linestyles[j], color='black')
            elif j > 3:
                ax.plot(sgrid, n_ss[:, j], label=this_label,
                        marker=markers[j - 4], color='black')
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'labor supply $n_{j,s}$')
        output_path = os.path.join(output_dir, 'n_ss_2D')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot 3D steady-state savings/wealth distribution
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'ability-$j$')
        ax.set_zlabel(r'indiv. savings $b_{j,s}$')
        ax.plot_surface(smat, jmat, b_ss.T, rstride=1,
                        cstride=6, cmap=cmap_c)
        output_path = os.path.join(output_dir, 'b_ss_3D')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot 2D steady-state savings/wealth distribution
        fig, ax = plt.subplots()
        linestyles = np.array(["-", "--", "-.", ":"])
        markers = np.array(["x", "v", "o", "d", ">", "|"])
        pct_lb = 0
        for j in range(J):
            this_label = (str(int(np.rint(pct_lb))) + " - " +
                          str(int(np.rint(pct_lb + 100 * lambdas[j]))) +
                          "%")
            pct_lb += 100 * lambdas[j]
            if j <= 3:
                ax.plot(sgrid, b_ss[:, j], label=this_label,
                        linestyle=linestyles[j], color='black')
            elif j > 3:
                ax.plot(sgrid, b_ss[:, j], label=this_label,
                        marker=markers[j - 4], color='black')
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'indiv. savings $b_{j,s}$')
        output_path = os.path.join(output_dir, 'b_ss_2D')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

    return ss_output
