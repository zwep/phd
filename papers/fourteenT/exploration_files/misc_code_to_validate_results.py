from objective_helper.fourteenT import VisualizeAllMetrics
from objective_configuration.fourteenT import CALC_OPTIONS, DDATA
from objective_helper.fourteenT import ReadMatData, OptimizeData
import numpy as np
import helper.misc as hmisc
import matplotlib.pyplot as plt

i_options = CALC_OPTIONS[0]
full_mask = i_options['full_mask']
type_mask = i_options['type_mask']
ddest = i_options['ddest']

coil_name_list = ['8 Channel Dipole Array 7T',
                     '8 Channel Dipole Array',
                     '16 Channel Loop Dipole Array',
                     '15 Channel Dipole Array',
                     '16 Channel Loop Array small',
                     '8 Channel Loop Array big']

sel_coil = '8 Channel Dipole Array'


sel_mat_file = sel_coil + "_ProcessedData.mat"  # zoiets...
mat_reader = ReadMatData(ddata=DDATA, mat_file=sel_mat_file)
data_obj = OptimizeData(mat_reader=mat_reader, normalization=normalization, full_mask=full_mask, type_mask=type_mask)
visual_obj = VisualizeAllMetrics(DDATA, ddest=ddest)

sel_x_metric = 'residual'
sel_y_metric = 'norm_power'


"""     Visualize current performance       """
# We have calculated metrics using `optimal` and `random` shims
# Check again if there is really a difference
# Also plot the lowest points of the point cloud of the `random` shims
v_optimal = visual_obj.optimal_json_data[sel_coil]
v_random = visual_obj.random_json_data[sel_coil]
new_x_coords, new_y_coords, new_min_coords = hmisc.get_minimum_curve(v_random[sel_x_metric], v_random[sel_y_metric])
fig, ax = plt.subplots()
ax.scatter(v_random[sel_x_metric], v_random[sel_y_metric], label=sel_coil, color='k', alpha=1)
ax.scatter(v_optimal[sel_x_metric], v_optimal[sel_y_metric], label=sel_coil, color='r', alpha=1)
ax.scatter(new_x_coords[1:], new_y_coords, label=sel_coil, color='y', alpha=1)
fig.suptitle(sel_coil)

# Here we see a difference...

"""     Re-do the calculation to get the optimal data... """
# Just to validate the optimal shims we have cacluated
v_optimal_again = data_obj.solve_trade_off()
fig, ax = plt.subplots()
ax.scatter(v_optimal[sel_x_metric], v_optimal[sel_y_metric], label=sel_coil, color='r', alpha=0.5)
ax.scatter(v_optimal_again[sel_x_metric], v_optimal_again[sel_y_metric], label=sel_coil, color='k', alpha=0.5)
fig.suptitle(sel_coil)

"""    Re calculate now with minimization stuff     """
v_optimal_mae = data_obj.solve_trade_off_mae(10)
fig, ax = plt.subplots()
ax.scatter(v_random[sel_x_metric], v_random[sel_y_metric], label=sel_coil, color='b', alpha=0.5)
ax.scatter(v_optimal[sel_x_metric], v_optimal[sel_y_metric], label=sel_coil, color='r', alpha=0.5)
ax.scatter(v_optimal_mae[sel_x_metric], v_optimal_mae[sel_y_metric], label=sel_coil, color='k', alpha=0.5)
fig.suptitle(sel_coil)

"""     Re-calculate the metrics that we are showing   """
# Get all shims, both random and optimal and recalculate the RMSE and norm solution
random_shim_array = np.array(v_random['random_shim'])[new_min_coords]
n_random = len(random_shim_array)
total_shim_array = np.concatenate([random_shim_array, v_optimal['opt_shim']])
selected_mask = data_obj.selected_mask
error_plot = []
norm_plot = []
for i_shim in total_shim_array:
    b1p_shimmed = np.abs((data_obj.mat_container['b1p'].T @ i_shim).T)
    rmse = np.sqrt(np.mean((b1p_shimmed[selected_mask == 1] - 1) ** 2))
    norm_shim = np.mean((np.abs(i_shim)) ** 2)
    error_plot.append(rmse)
    norm_plot.append(norm_shim)

plt.plot(error_plot)
plt.plot(norm_plot)

"""     Now try to solve everything with a different solver... """
# The idea is that we have some optimal solution in L2 sense
# Maybe with a different target function we DO get results that come close to the
# random shims...
# Because there should be a way to get to those..

def convert_to_cpx(x):
    n = len(x)
    x_real = x[:n // 2]
    x_imag = x[n // 2:]
    # return x_real + 1j * x_imag
    return x_real * np.exp(1j * x_imag)


def objective_mae(x):
    x_cpx = convert_to_cpx(x)
    system_shimmed = np.abs(data_obj.system_matrix @ x_cpx)
    mae = np.mean(np.abs(system_shimmed - 1)) + lambda_value * np.mean(np.abs(x_cpx) ** 2)
    return mae


def objective_rmse(x):
    global lambda_value
    x_cpx = convert_to_cpx(x)
    system_shimmed = np.abs(data_obj.system_matrix @ x_cpx)
    mae = np.sqrt(np.mean((system_shimmed - 1) ** 2)) + lambda_value * np.mean(np.abs(x_cpx) ** 2)
    return mae


import scipy.optimize

x_init = np.concatenate([np.ones(mat_reader.n_ports), np.zeros(mat_reader.n_ports)])

mae_solutions = []
for lambda_value in data_obj.lambda_range:
    x_opt_mae = scipy.optimize.minimize(fun=objective_mae, x0=x_init, tol=1e-8, method='CG')
    mae_solutions.append(x_opt_mae)

solutions_cpx_mae = [convert_to_cpx(x.x) for x in mae_solutions]
result_dict_list_mae = [data_obj.get_result_container(x) for x in solutions_cpx_mae]
result_dict_list_mae = hmisc.listdict2dictlist(result_dict_list_mae)

rmse_solutions = []
for lambda_value in data_obj.lambda_range:
    x_opt_rmse = scipy.optimize.minimize(fun=objective_rmse, x0=x_init, tol=1e-8, method='CG')
    rmse_solutions.append(x_opt_rmse)

solutions_cpx_rmse = [convert_to_cpx(x.x) for x in rmse_solutions]
result_dict_list_rmse = [data_obj.get_result_container(x) for x in solutions_cpx_rmse]
result_dict_list_rmse = hmisc.listdict2dictlist(result_dict_list_rmse)

# Solve closed form again..

data_obj.lambda_range = np.arange(0, 1, 0.1)
# Solve it again...
result_dict_list_closed = data_obj.solve_trade_off()

plt.figure()
plt.scatter(v_random[sel_x_metric], v_random[sel_y_metric], c='b')
plt.scatter(result_dict_list_closed[sel_x_metric], result_dict_list_closed[sel_y_metric], c='r')
plt.scatter(result_dict_list_mae[sel_x_metric], result_dict_list_mae[sel_y_metric], c='g')
plt.scatter(v_optimal[sel_x_metric], v_optimal[sel_y_metric], c='k')

plt.scatter(result_dict_list_rmse[sel_x_metric], result_dict_list_rmse[sel_y_metric], c='k')


"""
Temp"""


def heatmap(x, y, size):
    fig, ax = plt.subplots()

    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size * size_scale,  # Vector of square sizes, proportional to size parameter
        marker='s'  # Use square as scatterplot marker
    )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
