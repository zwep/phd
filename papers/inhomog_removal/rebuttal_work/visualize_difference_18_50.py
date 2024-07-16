import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

"""
Screw Excel... lets use matplotlib or python....

This is for single homog model

	    homogeneity			energy	
	    Resnet 18	Resnet 50		Resnet 18	Resnet 50
volunteer	1.03	1.07		2.8	2.76
patient_3T	0.14	0.09		0.2	0.17
patient	    0.19	0.15		0.27	0.19
synthetic	7.91	7.84		13.6	13.48

This is for multi bias field model

        homogeneity			energy	
        Resnet 18	Resnet 50		Resnet 18	Resnet 50
volunteer   1.02	1.02         2.78   2.79
synthetic	7.37    6.36       12.18     9.9

         coef of variation
         	Resnet 18	Resnet 50	
synthetic    0.02883827856618097           0.03158178981066539 (with target)
volunteer    -0.11620782470049337           -0.12780089946328566

                   
"""
FONTSIZE_XLABEL = 15
FONTSIZE_YLABEL = 15
FONTSIZE_XTICKS = 12
FONTSIZE_YTICKS = 12
FONTSIZE_TITLE = 20
FIG_SIZE = (10, 5)

def plot_homogeneity_energy(homogeneity_dict, energy_dict):
    fig, ax = plt.subplots(ncols=2, figsize=FIG_SIZE)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.8, wspace=None, hspace=None)
    fig.suptitle('Performance comparison between single-channel \n t-image model Resnet-18 and Resnet-50', fontsize=FONTSIZE_TITLE)
    ax_subplot = homogeneity_dict.plot.bar(ax=ax[0])
    ax_subplot.tick_params(axis='x', labelrotation=10, labelsize=FONTSIZE_XTICKS)
    ax_subplot.tick_params(axis='y', labelrotation=0, labelsize=FONTSIZE_YTICKS)
    # ax_subplot.set_title('Relative homogeneity for single-channel t-Image model')
    ax_subplot.set_ylabel('Relative homogeneity', fontsize=FONTSIZE_YLABEL)
    ax_subplot.set_xlabel('Datasets', fontsize=FONTSIZE_XLABEL)

    ax_subplot = energy_dict.plot.bar(ax=ax[1])
    ax_subplot.tick_params(axis='x', labelrotation=10, labelsize=FONTSIZE_XTICKS)
    ax_subplot.tick_params(axis='y', labelrotation=0, labelsize=FONTSIZE_YTICKS)
    ax_subplot.set_ylabel('Relative energy', fontsize=FONTSIZE_YLABEL)
    ax_subplot.set_xlabel('Datasets', fontsize=FONTSIZE_XLABEL)
    # ax_subplot.set_title('Relative energy for single-channel t-Image model')
    return fig


"""
Compare single channel homog model...
"""
resnet_18_t_image = {'homogeneity': {'volunteer': 1.03, 'patient 3T': 0.14, 'patient 7T': 0.19, 'synthetic': 7.91},
                     'energy': {'volunteer': 2.80, 'patient 3T': 0.2, 'patient 7T': 0.27, 'synthetic': 13.60}}
resnet_50_t_image = {'homogeneity': {'volunteer': 1.07, 'patient 3T': 0.09, 'patient 7T': 0.15, 'synthetic': 7.84},
                     'energy': {'volunteer': 2.76, 'patient 3T': 0.17, 'patient 7T': 0.19, 'synthetic': 13.48}}

compare_rel_homogeneity = pd.DataFrame({'resnet18': resnet_18_t_image['homogeneity'],
                                        'resnet50': resnet_50_t_image['homogeneity']})
compare_rel_energy = pd.DataFrame({'resnet18': resnet_18_t_image['energy'],
                                   'resnet50': resnet_50_t_image['energy']})

plot_homogeneity_energy(energy_dict=compare_rel_energy, homogeneity_dict=compare_rel_homogeneity)

"""
Compare multi channel biasfield model...
"""

resnet_18_t_biasf = {'homogeneity': {'volunteer': 1.02, 'synthetic': 7.37},
                   'energy': {'volunteer': 2.78, 'synthetic': 12.18}}
resnet_50_t_biasf = {'homogeneity': {'volunteer': 1.02, 'synthetic': 6.36},
                   'energy': {'volunteer': 2.79, 'synthetic': 9.9}}

compare_rel_homogeneity = pd.DataFrame({'resnet18': resnet_18_t_biasf['homogeneity'],
                                        'resnet50': resnet_50_t_biasf['homogeneity']})
compare_rel_energy = pd.DataFrame({'resnet18': resnet_18_t_biasf['energy'],
                                        'resnet50': resnet_50_t_biasf['energy']})


plot_homogeneity_energy(energy_dict=compare_rel_energy, homogeneity_dict=compare_rel_homogeneity)
