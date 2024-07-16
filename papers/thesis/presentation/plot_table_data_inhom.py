import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Data from the LaTeX table
models = [
    "Uncorrected",
    "N4 algorithm",
    "Single-channel t-Biasf",
    "Single-channel t-Image",
    "Multi-channel t-Biasf",
    "Multi-channel t-Image"
]

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

to_display = ["Uncorrected", "N4 algorithm", "Single-channel t-Image"]

metrics = ["WD", "SSIM", "RMSE"]
values = [
    [0.45, 0.23, 9.83],
    [0.44, 0.24, 9.91],
    [0.08, 0.76, 6.33],
    [0.06, 0.73, 6.38],
    [0.08, 0.77, 6.32],
    [0.06, 0.75, 6.28]
]

errors = [
    [0.03, 0.04, 0.65],
    [0.04, 0.05, 0.65],
    [0.02, 0.05, 0.54],
    [0.01, 0.04, 0.54],
    [0.02, 0.05, 0.56],
    [0.02, 0.04, 0.51]
]

# Number of models
n_models = len(models)

# Number of metrics
n_metrics = len(metrics)

# Set bar width
bar_width = 0.2

# Set positions for the bars
r = np.arange(len(metrics))

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
twinx_ax = ax.twinx()
bar_list = []

for ii, i_model_name in enumerate(to_display):
    i = models.index(i_model_name)
    _ = ax.bar(r[:-1] + ii * bar_width, values[i][:-1],
           width=bar_width,
           yerr=errors[i][:-1],
           capsize=5,
           label=i_model_name)
    _ = twinx_ax.bar(r[-1] + ii * bar_width, values[i][-1],
                  width=bar_width,
                  yerr=errors[i][-1],
                  capsize=5,
                  label=i_model_name)

# added these three lines
# labs = [l.get_label() for l in bar_list]
# ax.legend(bar_list, labs, loc=0)
ax.legend(loc='upper left')
# Adding labels
ax.set_ylabel('SSIM / WD value')
ax.set_ylim(0, 1)
twinx_ax.set_ylabel('RMSE value')
# ax.set_ylabel('Metric Value')
ax.set_title('Comparison of models across different metrics')
ax.set_xticks(r + bar_width)
ax.set_xticklabels([x for x in metrics], rotation=0, ha="right")


# Show plot
plt.tight_layout()
plt.show()
