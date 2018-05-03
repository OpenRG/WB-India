'''
------------------------------------------------------------------------
This module contains the functions that generate the variables
associated with households' optimization in the steady-state or in the
transition path of the overlapping generations model with S-period lived
agents and exogenous labor supply from Chapter 6 of the OG textbook.

This Python module imports the following module(s): None

This Python module defines the following function(s):
    get_cons()
    MU_c_stitch()
    get_b_errors()
                            bn_solve()
                            FOC_savings()
                            FOC_labor()
                            get_cnb_vecs()
                            c1_bSp1err()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
# import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_cons(bvec, b_cur, args):
    '''
    --------------------------------------------------------------------
    Calculate household consumption given prices, labor supply, current
    wealth, and savings
    --------------------------------------------------------------------
    INPUTS:
    bvec = scalar or (p-1, ) vector, savings over remaining periods of
           life
    b_cur = scalar, beginning period savings (individual state variable)
    args = length 3 tuple, (nvec, r, w)
    r     = scalar or (p,) vector, current interest rate or time path of
            interest rates over remaining life periods
    w     = scalar or (p,) vector, current wage or time path of wages
            over remaining life periods
    b     = scalar or (p,) vector, current period wealth or time path of
            current period wealths over remaining life periods
    b_sp1 = scalar or (p,) vector, savings for next period or time path
            of savings for next period over remaining life periods
    n     = scalar or (p,) vector, current labor supply or time path of
            labor supply over remaining life periods

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    nvec = scalar or (p,) vector, labor supply over remaining life
    r    = scalar > 0 or (p,) vector, interest rate over remaining life
    w    = scalar > 0 or (p,) vector, wage over remaining life
    cvec = scalar or (p,) vector, consumption over remaining life

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: cvec
    --------------------------------------------------------------------
    '''
    nvec, r, w = args
    b_s = np.append(b_cur, bvec)
    b_sp1 = np.append(bvec, 0.0)
    cvec = ((1 + r) * b_s) + (w * nvec) - b_sp1

    return cvec


def MU_c_stitch(cvec, sigma, graph=False):
    '''
    --------------------------------------------------------------------
    Generate marginal utility(ies) of consumption with CRRA consumption
    utility and stitched function at lower bound such that the new
    hybrid function is defined over all consumption on the real
    line but the function has similar properties to the Inada condition.

    u'(c) = c ** (-sigma) if c >= epsilon
          = g'(c) = 2 * b2 * c + b1 if c < epsilon

        such that g'(epsilon) = u'(epsilon)
        and g''(epsilon) = u''(epsilon)

        u(c) = (c ** (1 - sigma) - 1) / (1 - sigma)
        g(c) = b2 * (c ** 2) + b1 * c + b0
    --------------------------------------------------------------------
    INPUTS:
    cvec  = scalar or (p,) vector, individual consumption value or
            lifetime consumption over p consecutive periods
    sigma = scalar >= 1, coefficient of relative risk aversion for CRRA
            utility function: (c**(1-sigma) - 1) / (1 - sigma)
    graph = boolean, =True if want plot of stitched marginal utility of
            consumption function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    epsilon    = scalar > 0, positive value close to zero
    c_s        = scalar, individual consumption
    c_s_cnstr  = boolean, =True if c_s < epsilon
    b1         = scalar, intercept value in linear marginal utility
    b2         = scalar, slope coefficient in linear marginal utility
    MU_c       = scalar or (p,) vector, marginal utility of consumption
                 or vector of marginal utilities of consumption
    p          = integer >= 1, number of periods remaining in lifetime
    cvec_cnstr = (p,) boolean vector, =True for values of cvec < epsilon

    FILES CREATED BY THIS FUNCTION:
        MU_c_stitched.png

    RETURNS: MU_c
    --------------------------------------------------------------------
    '''
    epsilon = 0.0001
    if np.ndim(cvec) == 0:
        c_s = cvec
        c_s_cnstr = c_s < epsilon
        if c_s_cnstr:
            b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
            b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
            MU_c = 2 * b2 * c_s + b1
        else:
            MU_c = c_s ** (-sigma)
    elif np.ndim(cvec) == 1:
        p = cvec.shape[0]
        cvec_cnstr = cvec < epsilon
        MU_c = np.zeros(p)
        MU_c[~cvec_cnstr] = cvec[~cvec_cnstr] ** (-sigma)
        b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
        b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
        MU_c[cvec_cnstr] = 2 * b2 * cvec[cvec_cnstr] + b1

    if graph:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        cvec_CRRA   = (1000,) vector, support of c including values
                      between 0 and epsilon
        MU_CRRA     = (1000,) vector, CRRA marginal utility of
                      consumption
        cvec_stitch = (500,) vector, stitched support of consumption
                      including negative values up to epsilon
        MU_stitch   = (500,) vector, stitched marginal utility of
                      consumption
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        cvec_CRRA = np.linspace(epsilon / 2, epsilon * 3, 1000)
        MU_CRRA = cvec_CRRA ** (-sigma)
        cvec_stitch = np.linspace(-0.00005, epsilon, 500)
        MU_stitch = 2 * b2 * cvec_stitch + b1
        fig, ax = plt.subplots()
        plt.plot(cvec_CRRA, MU_CRRA, ls='solid', label='$u\'(c)$: CRRA')
        plt.plot(cvec_stitch, MU_stitch, ls='dashed', color='red',
                 label='$g\'(c)$: stitched')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Marginal utility of consumption with stitched ' +
                  'function', fontsize=14)
        plt.xlabel(r'Consumption $c$')
        plt.ylabel(r'Marginal utility $u\'(c)$')
        plt.xlim((-0.00005, epsilon * 3))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, "MU_c_stitched")
        plt.savefig(output_path)
        # plt.show()

    return MU_c


def get_b_errors(cvec, args):
    '''
    --------------------------------------------------------------------
    Generates vector of dynamic Euler errors that characterize the
    optimal lifetime savings decision. Because this function is used for
    solving for lifetime decisions in both the steady-state and in the
    transition path, lifetimes will be of varying length. Lifetimes in
    the steady-state will be S periods. Lifetimes in the transition path
    will be p in [2, S] periods
    --------------------------------------------------------------------
    INPUTS:
    cvec =
    args = length 4 tuple, (beta, sigma, r, diff)
    beta    = scalar in (0,1), discount factor
    sigma   = scalar > 0, coefficient of relative risk aversion
    r       = scalar > 0 or (p,) vector, steady-state interest rate or
              time path of interest rates
    cvec    = (p,) vector, distribution of consumption by age c_p
    diff    = boolean, =True if use simple difference Euler
              errors. Use percent difference errors otherwise.

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        MU_c_stitch()

    OBJECTS CREATED WITHIN FUNCTION:
    mu_c     = (p-1,) vector, marginal utility of current consumption
    mu_cp1   = (p-1,) vector, marginal utility of next period consumpt'n
    b_errors = (p-1,) vector, Euler errors characterizing optimal
               savings bvec

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_errors
    --------------------------------------------------------------------
    '''
    beta, sigma, r, diff = args
    mu_c = MU_c_stitch(cvec[:-1], sigma)
    mu_cp1 = MU_c_stitch(cvec[1:], sigma)

    if diff:
        b_errors = (beta * (1 + r) * mu_cp1) - mu_c
    else:
        b_errors = ((beta * (1 + r) * mu_cp1) / mu_c) - 1

    return b_errors


# def bn_solve(guesses, *args):
#     '''
#     --------------------------------------------------------------------
#     Finds the euler errors for certain b and n, one ability type at a time.
#     --------------------------------------------------------------------
#     INPUTS:
#     guesses = (2S-1,) vector, initial guesses for b and n
#     params  = length 10 tuple, r, w, S, beta, sigma, l_tilde, b_ellip,
#               upsilon, chi_n_vec, diff

#     OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
#         FOC_savings()
#         FOC_labor()

#     OBJECTS CREATED WITHIN FUNCTION:
#     r         = scalar, real interest rate
#     w         = scalar, real wage rate
#     S         =
#     beta      =
#     sigma     =
#     l_tilde   =
#     b_ellip   =
#     upsilon   =
#     chi_n_vec =
#     diff      =
#     b_guess   = [S,] vector, initial guess at household savings
#     n_guess   = [S,] vector, initial guess at household labor supply
#     b_s       = [S,] vector, wealth enter period with
#     b_splus1  = [S,] vector, household savings
#     b_splus2  = [S,] vector, household savings one period ahead
#     b_params  =
#     error1    = [S,] vector, errors from FOC for savings
#     n_params  =
#     error2    = [S,] vector, errors from FOC for labor supply
#     tax1 = [S,] vector, total income taxes paid
#     cons = [S,] vector, household consumption
#     RETURNS: 2Sx1 list of euler errors
#     OUTPUT: None
#     --------------------------------------------------------------------
#     '''
#     (r, w, S, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec,
#         diff) = args
#     b_guess = np.append(np.array(guesses[:S - 1]), 0.0)
#     n_guess = np.array(guesses[S - 1:])
#     b_s = np.array([0] + list(b_guess[:-1]))
#     b_splus1 = b_guess

#     cons = get_cons(r, w, b_s, b_splus1, n_guess)
#     b_params = (beta, sigma)
#     error1 = get_b_errors(b_params, r, cons, diff)

#     n_args = (w, cons, sigma, l_tilde, chi_n_vec, b_ellip, upsilon, diff)
#     error2 = get_n_errors(n_guess, n_args)

#     return list(error1.flatten()) + list(error2.flatten())


# def get_cnb_vecs(c_init, rpath, wpath, params):
#     '''
#     --------------------------------------------------------------------
#     Generate lifetime consumption vector for individual given a guess
#     for initial consumption c_{S-p+1}, initial wealth b_{S-p+1}, a path
#     for interest rates, wages, and parameters beta and sigma using
#     household first order equations

#     c_{s+1,t+1} = c_{s,t} * ([beta * (1 + r_{t+1})] ** (1 / sigma))

#     w_t * (c_{s,t} ** (-sigma)) = chi_n_s * g'(n_{s,t})
#     --------------------------------------------------------------------
#     INPUTS:
#     c_init = scalar > 0, consumption in initial period of lifetime c_{S-p+1}
#     rpath  = (p,) vector, path of interest rates over lifetime
#     wpath  = (p,) vector, path of wages over lifetime
#     params = length 8 tuple, (b_init, beta, sigma, l_tilde, b_ellip,
#              upsilon, chi_n_vec, diff)

#     OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
#         get_n_s()

#     OBJECTS CREATED WITHIN FUNCTION:
#     b_init    = scalar, initial wealth b_{S-p+1}
#     beta      = scalar in (0, 1), discount factor
#     sigma     = scalar >= 1, coefficient of relative risk aversion
#     b_ellip   = scalar > 0, fitted value of b for elliptical disutility
#                 of labor
#     l_tilde   = scalar > 0, per-period time endowment for every agent
#     upsilon   = scalar > 1, fitted value of upsilon for elliptical
#                 disutility of labor
#     chi_n_vec = (p,) vector, values for chi^n_s over remaining lifetime
#     diff      = boolean, =True if simple difference Euler error,
#                 otherwise percent deviation Euler error
#     p         = integer >= 1, number of periods remaining in a
#                 household's life
#     cvec      = (p,) vector, household lifetime consumption given c1
#     nvec      = (p,) vector, household lifetime labor supply given c1
#     bvec      = (p,) vector, household lifetime savings given c_{S-p+1}
#                 and b_{S-p+1} where b1=0
#     per       = integer >= 1, index of period number
#     n_args    = length 8 tuple, (c_s, w_t, sigma, l_tilde, chi_n_s,
#                 b_ellip, upsilon, diff)
#     n_options = length 1 dict, options for opt.root(get_n_s,...)
#     result_n  = results object, solution from opt.root(get_n_s,...)
#     b_Sp1     = scalar, savings after the last period of life. Should be
#                 zero in equilibrium

#     FILES CREATED BY THIS FUNCTION: None

#     RETURNS: cvec, nvec, bvec, b_Sp1
#     --------------------------------------------------------------------
#     '''
#     (b_init, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec,
#         diff) = params
#     p = rpath.shape[0]
#     cvec = np.zeros(p)
#     nvec = np.zeros(p)
#     bvec = np.zeros(p)
#     for per in range(p):
#         if per == 0:
#             bvec[per] = b_init
#             cvec[per] = c_init
#         else:
#             bvec[per] = ((1 + rpath[per - 1]) * bvec[per - 1] +
#                          wpath[per - 1] * nvec[per - 1] - cvec[per - 1])
#             cvec[per] = cvec[per - 1] * ((beta * (1 + rpath[per])) **
#                                          (1 / sigma))
#         n_options = {'maxiter': 500}
#         n_args = [wpath[per], cvec[per], sigma, l_tilde, chi_n_vec[per],
#                   b_ellip, upsilon, diff]
#         result_n = \
#             opt.root(get_n_errors, l_tilde / 2, args=(n_args),
#                      method='lm', tol=1e-14, options=(n_options))

#         nvec[per] = result_n.x
#     b_Sp1 = (1 + rpath[-1]) * bvec[-1] + wpath[-1] * nvec[-1] - cvec[-1]

#     return cvec, nvec, bvec, b_Sp1


# def c1_bSp1err(c_init, *args):
#     '''
#     --------------------------------------------------------------------
#     Given value for c1, as well as w and r, solve for household
#     lifetime decisions c_{s,t}, n_{s,t}, and b_{s+1,t+1}, and return
#     implied savings for period after last period of life b_{S+1}. This
#     savings amount b_{S+1} should be zero in equilibrium.
#     --------------------------------------------------------------------
#     INPUTS:
#     c_init = scalar > 0, assumed initial period consumption for
#              individual
#     args   = length 10 tuple, (b_init, beta, sigma, l_tilde, b_ellip,
#              upsilon, chi_n_vec, rpath, wpath, diff)

#     OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
#         get_cnb_vecs()

#     OBJECTS CREATED WITHIN FUNCTION:
#     beta      = scalar in (0, 1), discount factor
#     sigma     = scalar >= 1, coefficient of relative risk aversion
#     b_ellip   = scalar > 0, fitted value of b for elliptical disutility
#                 of labor
#     l_tilde   = scalar > 0, per-period time endowment for every agent
#     upsilon   = scalar > 1, fitted value of upsilon for elliptical
#                 disutility of labor
#     chi_n_vec = (p,) vector, values for chi^n_s for remaining lifetime
#     rpath     = (p,) vector, path of interest rates over remaining life
#     wpath     = (p,) vector, path of wages over remaining lifetime
#     diff      = boolean, =True if simple difference Euler errors,
#                 otherwise percent deviation Euler errors
#     cnb_args  = length 8 tuple, args to pass into get_cnb_vecs()
#     cvec      = (p,) vector, household lifetime consumption given c1
#     nvec      = (p,) vector, household lifetime labor supply given c1
#     bvec      = (p,) vector, household lifetime savings given c1
#                 (b_{S-p+1}, b_{S-p+2}, ...bS) where b1=0
#     b_Sp1     = scalar, residual amount left for individual savings in
#                 period after last period of life based on c_{S-p+1},
#                 Euler equations, and budget constraints. b_Sp1 should be
#                 zero in equilibrium

#     FILES CREATED BY THIS FUNCTION: None

#     RETURNS: b_Sp1
#     --------------------------------------------------------------------
#     '''
#     (b_init, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec, rpath,
#         wpath, diff) = args
#     cnb_args = (b_init, beta, sigma, l_tilde, b_ellip, upsilon,
#                 chi_n_vec, diff)
#     cvec, nvec, bvec, b_Sp1 = get_cnb_vecs(c_init, rpath, wpath,
#                                            cnb_args)

#     return b_Sp1
