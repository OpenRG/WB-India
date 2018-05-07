'''
------------------------------------------------------------------------
This module contains the functions that generate aggregate variables in
the steady-state or in the transition path of the overlapping
generations model with S-period lived agents and endogenous labor supply
from Chapter 7 of the OG textbook.

This Python module imports the following module(s): None

This Python module defines the following function(s):
    get_L()
    get_K()
    get_Y()
    get_C()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_L(narr, emat, lambdas):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate labor L or time path of aggregate
    labor L_t.

    We have included a stitching function for L when L<=epsilon such
    that the the adjusted value is the following. Let
    sum(lambda_j*e_{j,s}*n_{n,s,t}) = X. Market clearing is usually
    given by L = X:

    L = X when X >= epsilon
      = f(X) = a * exp(b * X) when X < epsilon
                   where a = epsilon / e  and b = 1 / epsilon

    This function has the properties that
    (i) f(X)>0 and f'(X) > 0 for all X,
    (ii) f(eps) = eps (i.e., functions X and f(X) meet at epsilon)
    (iii) f'(eps) = 1 (i.e., slopes of X and f(X) are equal at epsilon)
    --------------------------------------------------------------------
    INPUTS:
    narr    = (S, J) matrix or (S, J, T+S-1) array, values for steady-
              state labor supply (n_{j,s}) or time path of the
              distribution of labor supply (n_{j,s,t})
    emat    = (S, J) matrix, e_{j,s} ability by age and ability type
    lambdas = (J,) vector, income percentiles for ability types

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    epsilon    = scalar > 0, small value at which stitch f(X) function
    a          = scalar > 0, multiplicative factor in f(X) = a*exp(b*X)
    b          = scalar, multiplicative factor in exponent of
                 f(X) = a*exp(b*X)
    S          = integer >= 3, number of periods in individual life
    Tpers      = integer > 3, number of time periods in the time path
    J          = integer >= 1, number of ability types
    emat_arr   = (S, J, T+S-1) array, ability matrix copied across T+S-1
                 time periods
    lambda_mat = (S, J) matrix, lambdas vector copied down S rows
    lambda_arr = (S, J, T+S-1) array, lambda_mat matrix copied across
                   T+S-1 time periods
    L          = scalar > 0, aggregate labor
    L_cstr     = boolean or (T+S-1) boolean vector, =True if L < eps or
                 if L_t < eps

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: L
    --------------------------------------------------------------------
    '''
    epsilon = 0.1
    a = epsilon / np.exp(1)
    b = 1 / epsilon
    S = narr.shape[0]
    if narr.ndim == 2:  # This is the steady-state case
        lambda_mat = np.tile(lambdas, (S, 1))
        L = (narr * emat * lambda_mat).sum()
        L_cstr = L < epsilon
        if L_cstr:
            print('get_L() warning: distribution of labor supply ' +
                  'and/or parameters created L<epsilon')
            # Force L >= eps by stitching a * exp(b * L) for L < eps
            L = a * np.exp(b * L)
    elif narr.ndim == 3:  # This is the time path case
        Tpers = narr.shape[2]
        J = lambdas.shape[0]
        emat_arr = np.tile(np.reshape(emat, (S, J, 1)), (1, 1, Tpers))
        lambda_arr = np.tile(np.reshape(lambdas, (1, J, 1)),
                             (S, 1, Tpers))
        L = ((narr * emat_arr * lambda_arr).sum(0)).sum(0)
        L_cstr = L < epsilon
        if L.min() < epsilon:
            print('Aggregate labor constraint is violated ' +
                  '(L_t < epsilon) for some period in time path.')
            L[L_cstr] = a * np.exp(b * L[L_cstr])

    return L


def get_K(barr, lambdas):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate capital stock K or time path of
    aggregate capital stock K_t.

    We have included a stitching function for K when K<=epsilon such
    that the the adjusted value is the following. Let sum(b_i) = X.
    Market clearing is usually given by K = X:

    K = X when X >= epsilon
      = f(X) = a * exp(b * X) when X < epsilon
                   where a = epsilon / e  and b = 1 / epsilon

    This function has the properties that
    (i) f(X)>0 and f'(X) > 0 for all X,
    (ii) f(eps) = eps (i.e., functions X and f(X) meet at epsilon)
    (iii) f'(eps) = 1 (i.e., slopes of X and f(X) are equal at epsilon)
    --------------------------------------------------------------------
    INPUTS:
    barr    = (S, J) matrix or (S, J, T+S-1) array, values for steady-
              state savings (b_{j,s}) or time path of the distribution
              of savings (b_{j,s,t})
    lambdas = (J,) vector, income percentiles for distribution of
                ability within each cohort

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    epsilon    = scalar > 0, small value at which stitch f(X) function
    a          = scalar > 0, multiplicative factor in f(X) = a*exp(b*X)
    b          = scalar, multiplicative factor in exponent of
                 f(X) = a*exp(b*X)
    S          = integer >= 3, number of periods in individual life
    Tpers      = integer > 3, number of time periods in the time path
    J          = integer >= 1, number of ability types
    lambda_mat = (S, J) matrix, lambdas vector copied down S rows
    lambda_arr = (S, J, T+S-1) array, lambda_mat matrix copied across
                 T+S-1 time periods
    K          = scalar or (T+S-1,) vector, steady-state aggregate
                 capital stock or time path of aggregate capital stock
    K_cstr     = boolean or (T+S-1) boolean vector, =True if K < eps or
                 if K_t < eps

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: K, K_cstr
    --------------------------------------------------------------------
    '''
    epsilon = 0.1
    a = epsilon / np.exp(1)
    b = 1 / epsilon
    S = barr.shape[0]
    if barr.ndim == 2:  # This is the steady-state case
        lambda_mat = np.tile(lambdas, (S, 1))
        K = (barr * lambda_mat).sum()
        K_cstr = K < epsilon
        if K_cstr:
            print('get_K() warning: distribution of savings and/or ' +
                  'parameters created K<epsilon')
            # Force K >= eps by stitching a * exp(b *K) for K < eps
            K = a * np.exp(b * K)

    elif barr.ndim == 3:  # This is the time path case
        Tpers = barr.shape[2]
        J = lambdas.shape[0]
        lambda_arr = np.tile(np.reshape(lambdas, (1, J, 1)),
                             (S, 1, Tpers))
        K = ((barr * lambda_arr).sum(0)).sum(0)
        K_cstr = K < epsilon
        if K.min() < epsilon:
            print('Aggregate capital constraint is violated ' +
                  '(K < epsilon) for some period in time path.')
            K[K_cstr] = a * np.exp(b * K[K_cstr])

    return K, K_cstr


def get_Y(K, L, params):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate output Y or time path of aggregate
    output Y_t
    --------------------------------------------------------------------
    INPUTS:
    K      = scalar > 0 or (T+S-2,) vector, aggregate capital stock
             or time path of the aggregate capital stock
    L      = scalar > 0 or (T+S-2,) vector, aggregate labor or time
             path of the aggregate labor
    params = length 2 tuple, production function parameters
             (A, alpha)
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0,1), capital share of income

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    Y = scalar > 0 or (T+S-2,) vector, aggregate output (GDP) or
        time path of aggregate output (GDP)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Y
    --------------------------------------------------------------------
    '''
    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))

    return Y


def get_C(carr, lambdas):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate consumption C or time path of
    aggregate consumption C_t
    --------------------------------------------------------------------
    INPUTS:
    carr    = (S, J) matrix or (S, J, T) matrix, distribution of
              consumption c_{j,s} in steady state or time path for the
              distribution of consumption c_{j,s,t}
    lambdas = (J,) vector, income percentiles for distribution of
               ability within each cohort

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    S          = integer >= 3, number of periods in individual life
    Tpers      = integer > 3, number of time periods in the time path
    J          = integer >= 1, number of ability types
    lambda_mat = (S, J) matrix, lambdas vector copied down S rows
    lambda_arr = (S, J, T+S-1) array, lambda_mat matrix copied across
                 T+S-1 time periods
    C          = scalar > 0 or (T,) vector, aggregate consumption or
                 time path of aggregate consumption

    Returns: C
    --------------------------------------------------------------------
    '''
    S = carr.shape[0]
    if carr.ndim == 2:  # This is the steady-state case
        lambda_mat = np.tile(lambdas, (S, 1))
        C = (carr * lambda_mat).sum()
    elif carr.ndim == 3:  # This is the time path case
        Tpers = carr.shape[2]
        J = lambdas.shape[0]
        lambda_arr = np.tile(np.reshape(lambdas, (1, J, 1)),
                             (S, 1, Tpers))
        C = ((carr * lambda_arr).sum(0)).sum(0)

    return C
