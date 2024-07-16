import numpy as np
import itertools
import json
import matplotlib.pyplot as plt
"""
I ran 1.5T and 3T images remotely.. Now I can visualize them here

also did the 7T locally.. can do them here
"""


ddata = '/home/bugger/Documents/paper/homogeneity_index/feature_dict_prostate_weighting.json'
with open(ddata, 'r') as f:
    temp = f.read()

feature_dict_3T_1p5T = json.loads(temp)


fig, ax = plt.subplots(3, 1)
counter = 0
main_counter = 0
for k, v in feature_dict_3T_1p5T.items():
    print(k, v.keys())
    # ax[main_counter].hist(feature_dict_7T[k], label=f'7T' + '--' + k, density=True, bins=128, alpha=0.5)
    for metric_label, metric_value in v.items():
        if 'cor' not in metric_label:
            # metric_value = list(itertools.chain(*metric_value))
            metric_value = [x[5] for x in metric_value]
            print(len(metric_value))
            # ax[main_counter].hist(metric_value, label=metric_label + '--' + k, density=True, bins=128, alpha=0.5)
            ax[main_counter].plot(metric_value, label=metric_label + '--' + k, alpha=0.5)
            ax[main_counter].legend()
            # counter += 1
            # counter = counter % len(v)
    main_counter += 1

fig.savefig('/home/bugger/Documents/paper/homogeneity_index/comparisson_3T_1p5T_hist.png')

"""
Visualize 7T stuff
"""

ddata = '/home/bugger/Documents/paper/homogeneity_index/metrics_on_7T.json'
with open(ddata, 'r') as f:
    temp = f.read()

feature_dict_7T = json.loads(temp)

fig, ax = plt.subplots(3, 1)
counter = 0
main_counter = 0
for k, v in feature_dict_7T.items():
    print(k, v.keys())
    for metric_label, metric_value in v.items():
        print(metric_label, len(metric_value))
        metric_value = [float(list(x.values())[0]) for x in metric_value]
        ax[main_counter].plot(metric_value, label=metric_label + '--' + k, alpha=0.5)
        ax[main_counter].legend()
        # counter += 1
        # counter = counter % len(v)
    main_counter += 1

# Scatter plots
fig, ax = plt.subplots(3, 1)
counter = 0
main_counter = 0
for k, v in feature_dict_7T.items():
    print(k)
    print(len(v['rho']), len(v['bias']), len(v['uncor']))
    v_uncor = list(v['uncor'].values())
    v_bias = list(v['bias'].values())
    v_rho = list(v['rho'].values())
    ax[main_counter].scatter(v_uncor, v_rho, label=f'{k} -- uncor vs rho')
    ax[main_counter].scatter(v_uncor, v_bias, label=f'{k} -- uncor vs biasf')
    ax[main_counter].legend()
    main_counter += 1