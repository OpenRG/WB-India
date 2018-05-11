'''
This function generates the output and comparisons from the GAIN Act
proposed tax reform dynamic simulation versus the current law baseline
using OG-USA
'''
import numpy as np
import pickle
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os


def print_latex_table(matrix, firstCol_list, decimals=2, pct=False,
                      pre_char=''):
    '''
    --------------------------------------------------------------------
    This function returns a multi-line string of LaTeX table output
    derived from original input matrix
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    '''
    if matrix.ndim == 1:
        matrix = matrix.reshape((1, matrix.shape[0]))
    elif matrix.ndim > 2:
        err_msg = ('LATEX TABLE ERROR: Input matrix has more than 2 ' +
                   'dimensions.')
        raise RuntimeError(err_msg)
    rows, cols = matrix.shape
    if len(firstCol_list) != rows:
        err_msg = ('LATEX TABLE ERROR: Number of elements in ' +
                   'firstCol_list is not equal to number of rows in ' +
                   'matrix.')
        raise RuntimeError(err_msg)
    if not pct:
        matrix_prec = np.around(matrix, decimals=decimals)
    else:
        matrix_prec = np.around(matrix * 100, decimals=decimals - 2)
    mat_string = ''
    for row in range(rows):
        for col in range(cols):
            if col == 0:
                mat_string += '$' + firstCol_list[row] + '$ & '
            if pct:
                mat_string += (pre_char +
                               '{:.2f}'.format(matrix_prec[row, col]))
                mat_string += '\%'
            else:
                mat_string += (pre_char +
                               '{:.4f}'.format(matrix_prec[row, col]))
            if col < cols - 1:
                mat_string += ' & '
            if col == cols - 1:
                mat_string += ' \\\\ \n '
    print(mat_string)

    return mat_string


def print_latex_table2(matrix, firstCol_list, decimals=2, pct=False,
                       pre_char=''):
    '''
    --------------------------------------------------------------------
    This function returns a multi-line string of LaTeX table output
    derived from original input matrix
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    '''
    if matrix.ndim == 1:
        matrix = matrix.reshape((1, matrix.shape[0]))
    elif matrix.ndim > 2:
        err_msg = ('LATEX TABLE ERROR: Input matrix has more than 2 ' +
                   'dimensions.')
        raise RuntimeError(err_msg)
    rows, cols = matrix.shape
    if len(firstCol_list) != rows:
        err_msg = ('LATEX TABLE ERROR: Number of elements in ' +
                   'firstCol_list is not equal to number of rows in ' +
                   'matrix.')
        raise RuntimeError(err_msg)
    if not pct:
        matrix_prec = np.around(matrix, decimals=decimals)
    else:
        matrix_prec = np.around(matrix * 100, decimals=decimals - 2)
    mat_string = ''
    for row in range(rows):
        for col in range(cols):
            if col == 0:
                mat_string += firstCol_list[row] + ' & '
            if pct:
                mat_string += (pre_char +
                               '{:.2f}'.format(matrix_prec[row, col]))
                mat_string += '\%'
            else:
                mat_string += (pre_char +
                               '{:.4f}'.format(matrix_prec[row, col]))
            if col < cols - 1:
                mat_string += ' & '
            if col == cols - 1:
                mat_string += ' \\\\ \n '
    print(mat_string)

    return mat_string


# Check the tax functions from baseline to reform
tfuncs_bas_path = ('./OUTPUT_BASELINE_CL/TxFuncEst_baseline.pkl')
tfuncs_bas = pickle.load(open(tfuncs_bas_path, 'rb'))
avg_etr_bas = tfuncs_bas['tfunc_avg_etr']
avg_mtrx_bas = tfuncs_bas['tfunc_avg_mtrx']
avg_mtry_bas = tfuncs_bas['tfunc_avg_mtry']
tfuncs_ref_path = ('./OUTPUT_REFORM_CL_0/TxFuncEst_policy0.pkl')
tfuncs_ref = pickle.load(open(tfuncs_ref_path, 'rb'))
avg_etr_ref = tfuncs_ref['tfunc_avg_etr']
avg_mtrx_ref = tfuncs_ref['tfunc_avg_mtrx']
avg_mtry_ref = tfuncs_ref['tfunc_avg_mtry']

avg_etr_dif = avg_etr_ref - avg_etr_bas
avg_mtrx_dif = avg_mtrx_ref - avg_mtrx_bas
avg_mtry_dif = avg_mtry_ref - avg_mtry_bas

print('ETR baseline, reform, difference')
firstCol_ETR = ['$ETR$ baseline', '$ETR$ reform',
                '$ETR$ diff.\\tnote{a}']
ETR_mat = np.vstack((avg_etr_bas, avg_etr_ref, avg_etr_dif))
ETR_mat_latex = print_latex_table2(ETR_mat, firstCol_ETR, decimals=4,
                                   pct=True)
print('MTRx baseline, reform, difference')
firstCol_MTRx = ['$MTRx$ baseline', '$MTRx$ reform',
                 '$MTRx$ diff.\\tnote{a}']
MTRx_mat = np.vstack((avg_mtrx_bas, avg_mtrx_ref, avg_mtrx_dif))
MTRx_mat_latex = print_latex_table2(MTRx_mat, firstCol_MTRx, decimals=4,
                                    pct=True)
print('MTRy baseline, reform, difference')
firstCol_MTRy = ['$MTRy$ baseline', '$MTRy$ reform',
                 '$MTRy$ diff.\\tnote{a}']
MTRy_mat = np.vstack((avg_mtry_bas, avg_mtry_ref, avg_mtry_dif))
MTRy_mat_latex = print_latex_table2(MTRy_mat, firstCol_MTRy, decimals=4,
                                    pct=True)

# Import baseline equilibrium time paths
tpi_vars_bas_path = ('./OUTPUT_BASELINE_CL/TPI/TPI_vars.pkl')
tpi_vars_bas = pickle.load(open(tpi_vars_bas_path, 'rb'))
Ypath_bas = tpi_vars_bas['Y']
Cpath_bas = tpi_vars_bas['C']
Ipath_bas = tpi_vars_bas['I']
Kpath_bas = tpi_vars_bas['K']
Lpath_bas = tpi_vars_bas['L']
wpath_bas = tpi_vars_bas['w']
rpath_bas = tpi_vars_bas['r']
revpath_bas = tpi_vars_bas['REVENUE']
Dpath_bas = tpi_vars_bas['D']
Gpath_bas = tpi_vars_bas['G']
DYpath_bas = Dpath_bas[:-1] / Ypath_bas
GYpath_bas = Gpath_bas / Ypath_bas
c_mat_bas_5yr = tpi_vars_bas['c_path'][:5, :, :]
b_mat_bas_5yr = tpi_vars_bas['b_mat'][:5, :, :]
ati_mat_bas_5yr = c_mat_bas_5yr + b_mat_bas_5yr

# Import reform equilibrium time paths
tpi_vars_ref_path = ('./OUTPUT_REFORM_CL_0/TPI/TPI_vars.pkl')
# tpi_vars_ref = pickle.load(open(tpi_vars_ref_path, 'rb'), encoding='latin1')
tpi_vars_ref = pickle.load(open(tpi_vars_ref_path, 'rb'))
Ypath_ref = tpi_vars_ref['Y']
Cpath_ref = tpi_vars_ref['C']
Ipath_ref = tpi_vars_ref['I']
Kpath_ref = tpi_vars_ref['K']
Lpath_ref = tpi_vars_ref['L']
wpath_ref = tpi_vars_ref['w']
rpath_ref = tpi_vars_ref['r']
revpath_ref = tpi_vars_ref['REVENUE']
Dpath_ref = tpi_vars_ref['D']
Gpath_ref = tpi_vars_ref['G']
DYpath_ref = Dpath_ref[:-1] / Ypath_ref
GYpath_ref = Gpath_ref / Ypath_ref
c_mat_ref_5yr = tpi_vars_ref['c_path'][:5, :, :]
b_mat_ref_5yr = tpi_vars_ref['b_mat'][:5, :, :]
ati_mat_ref_5yr = c_mat_ref_5yr + b_mat_ref_5yr

# Create differences for plots
Ypath_pctdif = (Ypath_ref - Ypath_bas) / Ypath_bas
Cpath_pctdif = (Cpath_ref - Cpath_bas) / Cpath_bas
Ipath_pctdif = (Ipath_ref - Ipath_bas) / Ipath_bas
Kpath_pctdif = (Kpath_ref - Kpath_bas) / Kpath_bas
Lpath_pctdif = (Lpath_ref - Lpath_bas) / Lpath_bas
wpath_pctdif = (wpath_ref - wpath_bas) / wpath_bas
rpath_pctdif = (rpath_ref - rpath_bas) / rpath_bas
revpath_pctdif = (revpath_ref - revpath_bas) / revpath_bas
DYpath_dif = DYpath_ref - DYpath_bas
GYpath_dif = GYpath_ref - GYpath_bas
ati_pctch_5yr = (ati_mat_ref_5yr - ati_mat_bas_5yr) / ati_mat_bas_5yr
ati_pctch_5yr_vec = ati_pctch_5yr.flatten()
ati_pctch_5yr_sorted = ati_pctch_5yr_vec[np.argsort(ati_mat_bas_5yr,
                                                    axis=None)]
ati_pctch_5yravg_1qnt = ati_pctch_5yr_sorted[0:560].mean()
ati_pctch_5yravg_2qnt = ati_pctch_5yr_sorted[560:1120].mean()
ati_pctch_5yravg_3qnt = ati_pctch_5yr_sorted[1120:1680].mean()
ati_pctch_5yravg_4qnt = ati_pctch_5yr_sorted[1680:2240].mean()
ati_pctch_5yravg_5qnt = ati_pctch_5yr_sorted[2240:2800].mean()
# ati_pctch_5yravg_1qnt = np.sort(ati_pctch_5yr, axis=None)[0:560].mean()
# ati_pctch_5yravg_2qnt = np.sort(ati_pctch_5yr,
#                                 axis=None)[560:1120].mean()
# ati_pctch_5yravg_3qnt = np.sort(ati_pctch_5yr,
#                                 axis=None)[1120:1680].mean()
# ati_pctch_5yravg_4qnt = np.sort(ati_pctch_5yr,
#                                 axis=None)[1680:2240].mean()
# ati_pctch_5yravg_5qnt = np.sort(ati_pctch_5yr,
#                                 axis=None)[2240:2800].mean()
ati_pctch_5yravg_qnts = np.array([ati_pctch_5yravg_1qnt,
                                  ati_pctch_5yravg_2qnt,
                                  ati_pctch_5yravg_3qnt,
                                  ati_pctch_5yravg_4qnt,
                                  ati_pctch_5yravg_5qnt])
pickle.dump(ati_pctch_5yravg_qnts,
            open('ati_pctch_5yravg_qnts_cl.pkl', 'wb'))

print('Y, C, I, K, L pct chg, 10-years, 10-yr-avg, SS')
firstCol_YCIKL = ['Y_t', 'C_t', 'I_t', 'K_t', 'L_t']
YCIKL_mat = np.vstack((np.hstack((Ypath_pctdif[:10],
                                  np.mean(Ypath_pctdif[:10]),
                                  Ypath_pctdif[300])),
                       np.hstack((Cpath_pctdif[:10],
                                  np.mean(Cpath_pctdif[:10]),
                                  Cpath_pctdif[300])),
                       np.hstack((Ipath_pctdif[:10],
                                  np.mean(Ipath_pctdif[:10]),
                                  Ipath_pctdif[300])),
                       np.hstack((Kpath_pctdif[:10],
                                  np.mean(Kpath_pctdif[:10]),
                                  Kpath_pctdif[300])),
                       np.hstack((Lpath_pctdif[:10],
                                  np.mean(Lpath_pctdif[:10]),
                                  Lpath_pctdif[300]))
                       ))
YCIKL_mat_latex = print_latex_table(YCIKL_mat, firstCol_YCIKL,
                                    decimals=4, pct=True)
print('w, r pct chg, 10-years, 10-yr-avg, SS')
firstCol_wr = ['w_t', 'r_t']
wr_mat = np.vstack((np.hstack((wpath_pctdif[:10],
                               np.mean(wpath_pctdif[:10]),
                               wpath_pctdif[300])),
                    np.hstack((rpath_pctdif[:10],
                               np.mean(rpath_pctdif[:10]),
                               rpath_pctdif[300]))
                    ))
wr_mat_latex = print_latex_table(wr_mat, firstCol_wr, decimals=4,
                                 pct=True)
print('Rev, D/Y, G/Y pct chg (diffs), 10-years, 10-yr-avg, SS')
firstCol_RevDYGY = ['Rev_t', 'D_t/Y_t\\tnote{b}', 'G_t/Y_t\\tnote{b}']
revDYGY_mat = np.vstack((np.hstack((revpath_pctdif[:10],
                                    np.mean(revpath_pctdif[:10]),
                                    revpath_pctdif[300])),
                         np.hstack((DYpath_dif[:10],
                                    np.mean(DYpath_dif[:10]),
                                    DYpath_dif[300])),
                         np.hstack((GYpath_dif[:10],
                                    np.mean(GYpath_dif[:10]),
                                    GYpath_dif[300]))
                         ))
revDYGY_mat_latex = print_latex_table(revDYGY_mat, firstCol_RevDYGY,
                                      decimals=4, pct=True)

'''
------------------------------------------------------------------------
Plot differences
------------------------------------------------------------------------
cur_path    = string, path name of current directory
output_fldr = string, folder in current path to save files
output_dir  = string, total path of images folder
output_path = string, path of file name of figure to be saved
------------------------------------------------------------------------
'''
# Create directory if images directory does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
image_fldr = 'images'
image_dir = os.path.join(cur_path, image_fldr)
if not os.access(image_dir, os.F_OK):
    os.makedirs(image_dir)

plot_pers = int(60)
Gstab_per = int(20)
start_year = int(2018)
per_vec = np.arange(start_year, start_year + plot_pers)

# Figure 1: Y, C, I, K, and L percent changes
fig1, ax1 = plt.subplots()
plt.plot(per_vec, Ypath_pctdif[:plot_pers], '-', label='GDP ($Y_t$)')
plt.plot(per_vec, Cpath_pctdif[:plot_pers], '--',
         label='Consumption ($C_t$)')
# plt.plot(per_vec, Ipath_pctdif[:plot_pers], ':',
#          label='Investment ($I_t$)')
plt.plot(per_vec, Kpath_pctdif[:plot_pers], '-.',
         label='Capital Stock ($K_t$)')
plt.plot(per_vec, Lpath_pctdif[:plot_pers], '-', label='Labor ($L_t$)')
plt.axvline(x=start_year + Gstab_per, color='k')
# Shrink current axis by 20%
box_1 = ax1.get_position()
ax1.set_position([box_1.x0, box_1.y0, box_1.width * 0.75, box_1.height])
minorLocator_x = MultipleLocator(1)
minorLocator_y = MultipleLocator(0.01)
ax1.xaxis.set_minor_locator(minorLocator_x)
ax1.yaxis.set_minor_locator(minorLocator_y)
plt.grid(b=True, which='major', color='0.65', linestyle=':')
# plt.title('Time path of aggregate variables', fontsize=20)
plt.xlabel(r'Year $t$')
plt.xticks((2020, 2030, 2040, 2050, 2060, 2070, 2077),
           ('2020', '2030', '2040', '2050', '2060', '2070', '2077'))
plt.ylabel(r'Pct. change')
plt.yticks((-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01),
           ('-6%', '-5%', '-4%', '-3%', '-2%', '-1%', '0%', '1%'))
plt.xlim((start_year - 1, start_year + plot_pers))
plt.ylim((-0.061, 0.011))
plt.legend(bbox_to_anchor=(1.01, 0.8))
fig_path1 = os.path.join(image_dir, 'YCIKL_dif_GAIN_closed.png')
plt.savefig(fig_path1)
# plt.show()
plt.close()

# Figure 2: w and r percent changes
fig2, ax2 = plt.subplots()
plt.plot(per_vec, wpath_pctdif[:plot_pers], '-', label='wage ($w_t$)')
plt.plot(per_vec, rpath_pctdif[:plot_pers], '--',
         label='interest rate ($r_t$)')
plt.axvline(x=start_year + Gstab_per, color='k')
# Shrink current axis by 20%
box_2 = ax2.get_position()
ax2.set_position([box_2.x0, box_2.y0, box_2.width * 0.75, box_2.height])
minorLocator_x = MultipleLocator(1)
minorLocator_y = MultipleLocator(0.01)
ax2.xaxis.set_minor_locator(minorLocator_x)
ax2.yaxis.set_minor_locator(minorLocator_y)
plt.grid(b=True, which='major', color='0.65', linestyle=':')
# plt.title('Time path of aggregate variables', fontsize=20)
plt.xlabel(r'Year $t$')
plt.xticks((2020, 2030, 2040, 2050, 2060, 2070, 2077),
           ('2020', '2030', '2040', '2050', '2060', '2070', '2077'))
plt.ylabel(r'Pct. change')
plt.yticks((-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05),
           ('-2%', '-1%', '0%', '1%', '2%', '3%', '4%', '5%'))
plt.xlim((start_year - 1, start_year + plot_pers))
plt.ylim((-0.021, 0.051))
plt.legend(bbox_to_anchor=(1.48, 0.8))
fig_path2 = os.path.join(image_dir, 'wr_dif_GAIN_closed.png')
plt.savefig(fig_path2)
# plt.show()
plt.close()

# Figure 3: Revenue percent change, D/Y and G/Y differences
fig3, ax3 = plt.subplots()
plt.plot(per_vec, revpath_pctdif[:plot_pers], '-', label='Revenue')
plt.plot(per_vec, DYpath_dif[:plot_pers], '--',
         label='Debt/GDP (diff)')
plt.plot(per_vec, GYpath_dif[:plot_pers], ':',
         label='Spending/GDP (diff)')
plt.axvline(x=start_year + Gstab_per, color='k')
# Shrink current axis by 20%
box_3 = ax3.get_position()
ax3.set_position([box_3.x0, box_3.y0, box_3.width * 0.70, box_3.height])
minorLocator_x = MultipleLocator(1)
minorLocator_y = MultipleLocator(0.01)
ax3.xaxis.set_minor_locator(minorLocator_x)
ax3.yaxis.set_minor_locator(minorLocator_y)
plt.grid(b=True, which='major', color='0.65', linestyle=':')
# plt.title('Time path of aggregate variables', fontsize=20)
plt.xlabel(r'Year $t$')
plt.xticks((2020, 2030, 2040, 2050, 2060, 2070, 2077),
           ('2020', '2030', '2040', '2050', '2060', '2070', '2077'))
plt.ylabel(r'Pct. change (and diff.)')
plt.yticks((-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25),
           ('-10%', '-5%', '0%', '5%', '10%', '15%', '20%', '25%'))
plt.xlim((start_year - 1, start_year + plot_pers))
plt.ylim((-0.101, 0.251))
plt.legend(bbox_to_anchor=(1.59, 0.8))
fig_path3 = os.path.join(image_dir, 'RevDYGY_dif_GAIN_closed.png')
plt.savefig(fig_path3)
# plt.show()
plt.close()

# Figures 4a and 4b: Plot 5-year average (2018-2022) percent change in
# household labor supply by age and income
n_arr_bas = tpi_vars_bas['n_mat']
n_arr_bas_5 = n_arr_bas[:5, :, :]
n_arr_ref = tpi_vars_ref['n_mat']
n_arr_ref_5 = n_arr_ref[:5, :, :]
n_arr_pctch_5 = (n_arr_ref_5 - n_arr_bas_5) / n_arr_bas_5
n_arr_pctch_mean5 = n_arr_pctch_5.mean(axis=0)

ages = np.arange(21, 101)
J = n_arr_pctch_mean5.shape[1]
lambdas = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])
abil_midp = np.array([0.125, 0.375, 0.6, 0.75, 0.85, 0.94, 0.995])
abil_mesh, age_mesh = np.meshgrid(abil_midp, ages)

# 3D plot
cmap1 = cm.get_cmap('summer')
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(age_mesh, abil_mesh, n_arr_pctch_mean5, rstride=5,
                cstride=1, cmap=cmap1)
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'ability type-$j$')
ax.set_zlabel(r'Pct. chg. labor supply ($\%\Delta n_{j,s}$)')
fig_path4a = os.path.join(image_dir, 'n_pctch_3D_GAIN_closed.png')
plt.savefig(fig_path4a)
plt.close()

# 2D plot
fig = plt.figure()
ax = plt.subplot(111)
linestyles = np.array(['-', '--', '-.', ':'])
markers = np.array(['x', 'v', 'o', 'd', '>', '|'])
pct_lb = 0
for j in range(J):
    this_label = (str(int(np.rint(pct_lb))) + ' - ' +
                  str(int(np.rint(pct_lb +
                          100 * lambdas[j]))) + '%')
    pct_lb += 100 * lambdas[j]
    if j <= 3:
        ax.plot(ages, n_arr_pctch_mean5[:, j], label=this_label,
                linestyle=linestyles[j], color='black')
    elif j > 3:
        ax.plot(ages, n_arr_pctch_mean5[:, j], label=this_label,
                marker=markers[j - 4], color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
minorLocator_x = MultipleLocator(5)
minorLocator_y = MultipleLocator(0.00025)
ax.xaxis.set_minor_locator(minorLocator_x)
ax.yaxis.set_minor_locator(minorLocator_y)
plt.grid(b=True, which='major', color='0.65', linestyle=':')

ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.8))
plt.xlabel(r'age-$s$')
plt.xticks((20, 30, 40, 50, 60, 70, 80, 90, 100),
           ('20', '30', '40', '50', '60', '70', '80', '90', '100'))
plt.ylabel(r'Pct. chg. labor supply ($\%\Delta n_{j,s}$)')
plt.yticks((-0.0080, -0.0075, -0.0070, -0.0065, -0.0060, -0.0055,
            -0.0050, -0.0045, -0.0040),
           ('-0.80%', '-0.75%', '-0.70%', '-0.65%', '-0.60%', '-0.55%',
            '-0.50%', '-0.45%', '-0.40%'))
plt.ylim((-0.0081, -0.0039))
fig_path4b = os.path.join(image_dir, 'n_pctch_2D_GAIN_closed.png')
plt.savefig(fig_path4b)
plt.close()

# Figures 5a and 5b: Plot 5-year average (2018-2022) percent change in
# household savings by age and income
b_arr_bas = tpi_vars_bas['b_mat']
b_arr_bas_5 = b_arr_bas[:5, :, :]
b_arr_ref = tpi_vars_ref['b_mat']
b_arr_ref_5 = b_arr_ref[:5, :, :]
b_arr_pctch_5 = (b_arr_ref_5 - b_arr_bas_5) / b_arr_bas_5
b_arr_pctch_mean5 = b_arr_pctch_5.mean(axis=0)

# 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(age_mesh, abil_mesh, b_arr_pctch_mean5, rstride=5,
                cstride=1, cmap=cmap1)
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'ability type-$j$')
ax.set_zlabel(r'Pct. chg. savings ($\%\Delta b_{j,s}$)')
fig_path5a = os.path.join(image_dir, 'b_pctch_3D_GAIN_closed.png')
plt.savefig(fig_path5a)
plt.close()

# 2D plot
fig = plt.figure()
ax = plt.subplot(111)
pct_lb = 0
for j in range(J):
    this_label = (str(int(np.rint(pct_lb))) + ' - ' +
                  str(int(np.rint(pct_lb +
                          100 * lambdas[j]))) + '%')
    pct_lb += 100 * lambdas[j]
    if j <= 3:
        ax.plot(ages, b_arr_pctch_mean5[:, j], label=this_label,
                linestyle=linestyles[j], color='black')
    elif j > 3:
        ax.plot(ages, b_arr_pctch_mean5[:, j], label=this_label,
                marker=markers[j - 4], color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
minorLocator_x = MultipleLocator(5)
minorLocator_y = MultipleLocator(0.00025)
ax.xaxis.set_minor_locator(minorLocator_x)
ax.yaxis.set_minor_locator(minorLocator_y)
plt.grid(b=True, which='major', color='0.65', linestyle=':')

ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.8))
plt.xlabel(r'age-$s$')
plt.xticks((20, 30, 40, 50, 60, 70, 80, 90, 100),
           ('20', '30', '40', '50', '60', '70', '80', '90', '100'))
plt.ylabel(r'Pct. chg. savings ($\%\Delta b_{j,s}$)')
plt.yticks((0.0000, 0.0010, 0.0020, 0.0030, 0.0040, 0.0050),
           ('0.00%', '0.10%', '0.20%', '0.30%', '0.40%', '0.50%'))
plt.ylim((-0.0003, 0.0052))
fig_path5b = os.path.join(image_dir, 'b_pctch_2D_GAIN_closed.png')
plt.savefig(fig_path5b)
plt.close()

# Figures 6a and 6b: Plot 5-year average (2018-2022) percent change in
# household consumption by age and income
c_arr_bas = tpi_vars_bas['c_path']
c_arr_bas_5 = c_arr_bas[:5, :, :]
c_arr_ref = tpi_vars_ref['c_path']
c_arr_ref_5 = c_arr_ref[:5, :, :]
c_arr_pctch_5 = (c_arr_ref_5 - c_arr_bas_5) / c_arr_bas_5
c_arr_pctch_mean5 = c_arr_pctch_5.mean(axis=0)

# 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(age_mesh, abil_mesh, c_arr_pctch_mean5, rstride=5,
                cstride=1, cmap=cmap1)
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'ability type-$j$')
ax.set_zlabel(r'Pct. chg. consumption ($\%\Delta c_{j,s}$)')
fig_path6a = os.path.join(image_dir, 'c_pctch_3D_GAIN_closed.png')
plt.savefig(fig_path6a)
plt.close()

# 2D plot
fig = plt.figure()
ax = plt.subplot(111)
pct_lb = 0
for j in range(J):
    this_label = (str(int(np.rint(pct_lb))) + ' - ' +
                  str(int(np.rint(pct_lb +
                          100 * lambdas[j]))) + '%')
    pct_lb += 100 * lambdas[j]
    if j <= 3:
        ax.plot(ages, c_arr_pctch_mean5[:, j], label=this_label,
                linestyle=linestyles[j], color='black')
    elif j > 3:
        ax.plot(ages, c_arr_pctch_mean5[:, j], label=this_label,
                marker=markers[j - 4], color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
minorLocator_x = MultipleLocator(5)
minorLocator_y = MultipleLocator(0.00025)
ax.xaxis.set_minor_locator(minorLocator_x)
ax.yaxis.set_minor_locator(minorLocator_y)
plt.grid(b=True, which='major', color='0.65', linestyle=':')

ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.8))
plt.xlabel(r'age-$s$')
plt.xticks((20, 30, 40, 50, 60, 70, 80, 90, 100),
           ('20', '30', '40', '50', '60', '70', '80', '90', '100'))
plt.ylabel(r'Pct. chg. consumption ($\%\Delta c_{j,s}$)')
plt.yticks((0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035,
            0.0040, 0.0045),
           ('0.05%', '0.10%', '0.15%', '0.20%', '0.25%', '0.30%',
            '0.35%', '0.40%', '0.45%'))
plt.ylim((0.0002, 0.0047))
fig_path6b = os.path.join(image_dir, 'c_pctch_2D_GAIN_closed.png')
plt.savefig(fig_path6b)
plt.close()
