import pyensae
import re
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np
from pyensae.graphhelper import Corrplot
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import os

ddir_okt = '/home/bugger/Documents/data/data_from_rick/okt'
ddir_nov = '/home/bugger/Documents/data/data_from_rick/nov'
#
# """ Visualize Table 1"""
# data = pd.read_csv(os.path.join(ddir_okt, 'table_1.csv'))
# data.index = data.columns
# data = data.fillna(0)
# data = data.T + data
# np.fill_diagonal(data.values, 1)
# c = Corrplot(data)
#
# fig, ax = plt.subplots()
# ax_c, cb_obj = c.plot(fig=fig, figsize=(6, 6), method='circle', colorbar=True,
#                       cmap='seismic_r')
# scalarmap = cm.ScalarMappable(norm=mcolors.Normalize(*[-1, 1]), cmap='seismic_r')
# cb_obj.update_normal(scalarmap)
# cb_obj.set_ticks([-1, -0.5, 0, 0.5, 1])
# cb_obj.ax.tick_params(size=0)
# fig.savefig(os.path.join(ddir_okt, 'table_1.png'))
#
# """ Visualize Table 2"""
# data = pd.read_csv(os.path.join(ddir_okt, 'table_2.csv'))
# # data_col = [re.sub('Compactness', 'Compactness2' ,x) for x in data.columns]
# data.columns = data.columns
# data.index = data.columns
# data = data.fillna(0)
# data = data.T + data
# np.fill_diagonal(data.values, 1)
# c = Corrplot(data)
#
# fig, ax = plt.subplots()
# ax_c, cb_obj = c.plot(fig=fig, figsize=(6, 6), method='circle', colorbar=True,
#                       cmap='seismic_r')
# scalarmap = cm.ScalarMappable(norm=mcolors.Normalize(*[-1, 1]), cmap='seismic_r')
# cb_obj.update_normal(scalarmap)
# cb_obj.set_ticks([-1, -0.5, 0, 0.5, 1])
# cb_obj.ax.tick_params(size=0)
# fig.savefig(os.path.join(ddir_okt, 'table_2.png'))


""" Visualize 3"""
data = pd.read_csv(os.path.join(ddir_okt, 'table_3.csv'))

index_names = data['Unnamed: 0'].values
del data['Unnamed: 0']
data.index = index_names

fig, ax = plt.subplots()
c = Corrplot(data, compute_correlation=False)
ax_c, cb_obj = c.plot(fig=fig, figsize=(6, 6), method='circle', colorbar=True,
                      cmap='seismic_r', order_method=None)
scalarmap = cm.ScalarMappable(norm=mcolors.Normalize(*[-1, 1]), cmap='seismic_r')
cb_obj.update_normal(scalarmap)
cb_obj.set_ticks([-1, -0.5, 0, 0.5, 1])
cb_obj.ax.tick_params(size=0)
fig.savefig(os.path.join(ddir_okt, 'table_3.png'))


""" Visualize 4"""
data = pd.read_csv(os.path.join(ddir_nov, 'table_4.csv'), decimal=',')

index_names = data['Unnamed: 0'].values
del data['Unnamed: 0']
data.index = index_names

fig, ax = plt.subplots()
c = Corrplot(data.T, compute_correlation=False)
ax_c, cb_obj = c.plot(fig=fig, figsize=(6, 6), method='circle', colorbar=True,
                      cmap='seismic_r', order_method=None)
scalarmap = cm.ScalarMappable(norm=mcolors.Normalize(*[-1, 1]), cmap='seismic_r')
cb_obj.update_normal(scalarmap)
cb_obj.set_ticks([-1, -0.5, 0, 0.5, 1])
cb_obj.ax.tick_params(size=0)

fig.savefig(os.path.join(ddir_nov, 'table_4_T.png'))

sel_col = ['Elongation', 'Flatness', 'Major axis length',  'Area', 'Volume']

fig, ax = plt.subplots()
c = Corrplot(data[sel_col].T, compute_correlation=False)
ax_c, cb_obj = c.plot(fig=fig, figsize=(6, 6), method='circle', colorbar=True,
                      cmap='seismic_r', order_method=None)
scalarmap = cm.ScalarMappable(norm=mcolors.Normalize(*[-1, 1]), cmap='seismic_r')
cb_obj.update_normal(scalarmap)
cb_obj.set_ticks([-1, -0.5, 0, 0.5, 1])
cb_obj.ax.tick_params(size=0)


fig.savefig(os.path.join(ddir_nov, 'table_4_sel_col_T.png'))


""" Visualize 5"""
data = pd.read_csv(os.path.join(ddir_nov, 'table_5.csv'), decimal=',')

index_names = data['Unnamed: 0'].values
del data['Unnamed: 0']
data.index = index_names

# Nice.. this is a great example of how we copy references
data_transposed = data.T
data_transposed.index.values[-1] = 'Shape\n Index'


fig, ax = plt.subplots()
c = Corrplot(data, compute_correlation=False)
ax_c, cb_obj = c.plot(fig=fig, figsize=(6, 6), method='circle', colorbar=True,
                      cmap='seismic_r', order_method=None)
scalarmap = cm.ScalarMappable(norm=mcolors.Normalize(*[-1, 1]), cmap='seismic_r')
cb_obj.update_normal(scalarmap)
cb_obj.set_ticks([-1, -0.5, 0, 0.5, 1])
cb_obj.ax.tick_params(size=0)

fig.savefig(os.path.join(ddir_nov, 'table_5.png'))

exclude_col = ['minor axis length', 'least axis length']
sel_col = [x for x in data.columns if x.lower() not in exclude_col]

fig, ax = plt.subplots()
c = Corrplot(data[sel_col], compute_correlation=False)
ax_c, cb_obj = c.plot(fig=fig, figsize=(6, 6), method='circle', colorbar=True,
                      cmap='seismic_r', order_method=None)
scalarmap = cm.ScalarMappable(norm=mcolors.Normalize(*[-1, 1]), cmap='seismic_r')
cb_obj.update_normal(scalarmap)
cb_obj.set_ticks([-1, -0.5, 0, 0.5, 1])
cb_obj.ax.tick_params(size=0)


fig.savefig(os.path.join(ddir_nov, 'table_5_sel_col.png'))