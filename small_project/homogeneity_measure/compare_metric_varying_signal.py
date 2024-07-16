
import json
import matplotlib.pyplot as plt

"""
We ran stuff remotely

Lets chekc it out
"""

ddata = '/home/bugger/Documents/paper/homogeneity_index/varying_signal_feature_dict_train_data.json'
with open(ddata, 'r') as f:
    temp = f.read()

feature_dict = json.loads(temp)

fig, ax = plt.subplots(4)
counter = 0
for k, v in feature_dict.items():
    print(k, v.keys())
    for metric_label, metric_value in v.items():
        ax[counter].plot(metric_value, label=metric_label)
        ax[counter].legend()
        counter += 1
        counter = counter % len(v)



fig, ax = plt.subplots(3, figsize=(20, 15))
counter = 0
for k, v in feature_dict.items():
    print(k, v.keys())
    ax[0].scatter(v['ssim'], v['hi_integral'], color='b')
    ax[1].scatter(v['ssim'], v['fuzzy_luka_2'], color='r')
    ax[2].scatter(v['ssim'], v['glcm_contrast'], color='k')

ax[0].set_title('hi')
ax[1].set_title('fuzzy')
ax[2].set_title('glcm')
fig.savefig('/home/bugger/Documents/paper/homogeneity_index/varying_signal_feature_training_set.png')