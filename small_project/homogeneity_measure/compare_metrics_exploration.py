import scipy.stats
import json
import numpy as np
import matplotlib.pyplot as plt
import helper.misc as hmisc
import helper.plot_class as hplotc

"""
We have done some experiments with metrics... we see that HI is sitll vastly better htan the other

HI does have some problems... if multiple values are equal to 1 it LOOKS (according to HI) more 
homogeneous. But in reality these values (==1) can be random noise as well.
"""


def plot_relation(x, y, title=''):
    corr_value, p_value = scipy.stats.pearsonr(x, y)
    fig, ax = plt.subplots()
    plt.scatter(x, y)
    fig.suptitle(title + '\n Correlation    ' + str(np.round(corr_value, 2)))


ddict_hi = '/home/bugger/Documents/paper/homogeneity_index/metric_dict_hi.json'
ddict_fuzzy = '/home/bugger/Documents/paper/homogeneity_index/metric_dict_fuzzy.json'
ddict_glcm = '/home/bugger/Documents/paper/homogeneity_index/metric_dict_glcm.json'

hi_features = hmisc.load_json(ddict_hi)
fuzzy_features = hmisc.load_json(ddict_fuzzy)
glcm_features = hmisc.load_json(ddict_glcm)

"""
Lets reiterate what the metrics, compared to the SSIM, relate
"""

# Check the relation between the input homogeneity (HI)
# And the difference between target and input (SSIM)
x_var = hi_features['hi_integral_input']
y_var = hi_features['ssim_target_input']
plot_relation(x_var, y_var)

# Check the relation between the predicted homogeneity (HI)
# And the difference between target and prediction (SSIM)
x_var = hi_features['hi_integral_pred']
y_var = hi_features['ssim_target_pred']
plot_relation(x_var, y_var)

# Starting with the current HI... where does it move to
x_var = hi_features['hi_integral_input']
y_var = hi_features['hi_integral_pred']
fig, ax = plt.subplots(2)
ax[0].hist(x_var, bins=128, label='initial HI', alpha=0.5)
ax[0].hist(y_var, bins=128, color='r', label='pred HI', alpha=0.5)
plot_relation(x_var, y_var)

x_var = hi_features['ssim_target_input']
y_var = hi_features['ssim_target_pred']
plot_relation(x_var, y_var)
# ax[1].hist(x_var, bins=128, label='ssim target input', alpha=0.5)
# ax[1].hist(y_var, bins=128, color='r', label='ssim target pred', alpha=0.5)

"""
Now check the GLCM features
"""

# Check the relation between the input homogeneity (HI)
# And the difference between target and input (SSIM)
x_var = glcm_features['contrast_input']
y_var = glcm_features['ssim_target_input']
plot_relation(x_var, y_var)

x_var = glcm_features['contrast_pred']
y_var = glcm_features['ssim_target_pred']
plot_relation(x_var, y_var)

x_var = glcm_features['contrast_input']
y_var = glcm_features['contrast_pred']
fig, ax = plt.subplots(2)
ax[0].hist(x_var, bins=128, label='initial contrast', alpha=0.5)
ax[0].hist(y_var, bins=128, color='r', label='pred contrast', alpha=0.5)

"""
Now check the fuzzy features
"""

# Mean aggregate function on the input...
# Compared with target/input
x_var = fuzzy_features['luka_two_input']
y_var = fuzzy_features['ssim_target_input']
plot_relation(x_var, y_var, title='luka 2 input - ssim target/input')

x_var = fuzzy_features['luka_two_pred']
y_var = fuzzy_features['ssim_target_pred']
plot_relation(x_var, y_var, title='luka 2 pred - ssim target/pred')

# Starting with the current HI... where does it move to
x_var = fuzzy_features['luka_two_input']
y_var = fuzzy_features['luka_two_pred']
fig, ax = plt.subplots(2)
ax[0].hist(x_var, bins=128, label='initial Luka', alpha=0.5)
ax[0].hist(y_var, bins=128, color='r', label='pred Luka', alpha=0.5)
