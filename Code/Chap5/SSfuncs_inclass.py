import numpy as np
import os
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def get_K(bvec, lambdas):
    '''
    Compute the aggregate capital stock
    '''
    K = (bvec * lambdas).sum()
    return K


def get_C(cvec, lambdas):
    '''
    Compute the aggregate consumption
    '''
    C = (cvec * lambdas).sum()
    return C


def get_L(nvec, emat, lambdas):
    '''
    Compute aggregate labor
    '''
    L = (emat * nvec * lambdas).sum()

    return L


def get_r(L, K, A, alpha, delta):
    r = alpha * A * (L / K) ** (1 - alpha) - delta
    return r


def get_w_from_r(r, A, alpha, delta):
    w = (1 - alpha) * A * ((alpha * A) / (r + delta)) ** (alpha /
                                                          (1 - alpha))
    return w


def get_Y(K, L, A, alpha):
    Y = A * (K ** alpha) * (L ** (1 - alpha))
    return Y


def get_c(nvec, bvec, r, w, e):
    b_s = np.append(0.0, bvec)
    b_splus1 = np.append(bvec, 0.0)
    cvec = (1 + r) * b_s + w * e * nvec - b_splus1
    return cvec


def MDU_n(nvec, l_tilde, b_ellip, upsilon):
    MDU = ((b_ellip / l_tilde) * ((nvec / l_tilde) ** (upsilon - 1)) *
           ((1 - ((nvec / l_tilde) ** upsilon)) ** ((1 - upsilon) /
                                                    upsilon)))
    return MDU


def MU_c(cvec, sigma):
    MUc = cvec ** -sigma
    return MUc


def get_n_errors(nvec, bvec, r, w, args):
    sigma, chi_n_vec, l_tilde, b_ellip, upsilon, e = args
    MDU_ns = MDU_n(nvec, l_tilde, b_ellip, upsilon)
    c_s = get_c(nvec, bvec, r, w, e)
    MU_cs = MU_c(c_s, sigma)
    n_errors = w * e * MU_cs - chi_n_vec * MDU_ns
    return n_errors


def get_b_errors(nvec, bvec, r, w, args):
    beta, sigma, e = args
    cvec = get_c(nvec, bvec, r, w, e)
    c_s = cvec[:-1]
    c_sp1 = cvec[1:]
    MU_cs = MU_c(c_s, sigma)
    MU_csp1 = MU_c(c_sp1, sigma)
    b_errors = MU_cs - beta * (1 + r) * MU_csp1
    return b_errors


def euler_system(nbvec, *args):
    (r, w, beta, sigma, chi_n_vec, l_tilde, b_ellip, upsilon, S, e) = args
    nvec = nbvec[:S]
    bvec = nbvec[S:]
    n_args = (sigma, chi_n_vec, l_tilde, b_ellip, upsilon, e)
    n_errors = get_n_errors(nvec, bvec, r, w, n_args)
    b_args = (beta, sigma, e)
    b_errors = get_b_errors(nvec, bvec, r, w, b_args)
    nb_errors = np.append(n_errors, b_errors)
    return nb_errors


def SS_solve(nbvec_init, ss_args):
    (rbar, wbar, beta, sigma, chi_n_vec, l_tilde, b_ellip, upsilon,
     S, J, lambdas, emat, A, alpha, delta) = ss_args
    n_ss = np.empty((S, J))
    b_ss = np.empty((S - 1, J))
    n_errors_ss = np.empty((S, J))
    b_errors_ss = np.empty((S - 1, J))
    for j in range(J):
        nb_args = (rbar, wbar, beta, sigma, chi_n_vec, l_tilde, b_ellip,
                   upsilon, S, emat[:, j])
        results = opt.root(euler_system, nbvec_init, args=(nb_args),
                           method='lm', tol=1e-14)
        n_ss[:, j] = results.x[:S]
        b_ss[:, j] = results.x[S:]
        n_errors_ss[:, j] = results.fun[:S]
        b_errors_ss[:, j] = results.fun[S:]

    K_ss = get_K(b_ss, lambdas)
    L_ss = get_L(n_ss, emat, lambdas)
    r_ss = get_r(L_ss, K_ss, A, alpha, delta)
    w_ss = get_w_from_r(r_ss, A, alpha, delta)
    c_ss = np.empty((S, J))
    for j in range(J):
        c_ss[:, j] = get_c(n_ss[:, j], b_ss[:, j], r_ss, w_ss, emat[:, j])
    C_ss = get_C(c_ss, lambdas)
    Y_ss = get_Y(K_ss, L_ss, A, alpha)

    res_cnstr_errors = Y_ss - C_ss - delta * K_ss

    print('n_ss=', n_ss)
    print('b_ss=', b_ss)
    print('Max. abs. n_error=', np.absolute(n_errors_ss).max())
    print('Max. abs. b_error=', np.absolute(b_errors_ss).max())
    print('Max. abs. res. cnstr. error=',
          np.absolute(res_cnstr_errors).max())
    print('r_ss=', r_ss, 'w_ss=', w_ss, 'K_ss=', K_ss, 'L_ss=', L_ss,
          'Y_ss=', Y_ss)

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

    age_pers_c = np.arange(21, 21 + S)
    age_pers_b = np.arange(21, 21 + S - 1)

    # Plot steady-state consumption distribution
    fig, ax = plt.subplots()
    plt.plot(age_pers_c, c_ss[:, 0], marker='D', label='j = 1')
    plt.plot(age_pers_c, c_ss[:, 1], marker='D', label='j = 2')
    plt.plot(age_pers_c, c_ss[:, 2], marker='D', label='j = 3')
    plt.plot(age_pers_c, c_ss[:, 3], marker='D', label='j = 4')
    plt.plot(age_pers_c, c_ss[:, 4], marker='D', label='j = 5')
    plt.plot(age_pers_c, c_ss[:, 5], marker='D', label='j = 6')
    plt.plot(age_pers_c, c_ss[:, 6], marker='D', label='j = 7')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    # plt.title('Steady-state consumption', fontsize=20)
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Consumption')
    plt.xlim((20, 20 + S + 2))
    plt.legend(loc='upper right')
    output_path = os.path.join(output_dir, 'SS_c')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot steady-state labor supply distribution
    fig, ax = plt.subplots()
    plt.plot(age_pers_b, b_ss[:, 0], marker='D', label='j = 1')
    plt.plot(age_pers_b, b_ss[:, 1], marker='D', label='j = 2')
    plt.plot(age_pers_b, b_ss[:, 2], marker='D', label='j = 3')
    plt.plot(age_pers_b, b_ss[:, 3], marker='D', label='j = 4')
    plt.plot(age_pers_b, b_ss[:, 4], marker='D', label='j = 5')
    plt.plot(age_pers_b, b_ss[:, 5], marker='D', label='j = 6')
    plt.plot(age_pers_b, b_ss[:, 6], marker='D', label='j = 7')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    # plt.title('Steady-state savings', fontsize=20)
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Savings')
    plt.xlim((20, 20 + S + 2))
    plt.legend(loc='upper right')
    output_path = os.path.join(output_dir, 'SS_b')
    plt.savefig(output_path)
    # plt.show()
    plt.close()
