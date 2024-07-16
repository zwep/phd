import helper.misc as hmisc
from objective_configuration.reconstruction import DRESULT
import helper.plot_class as hplotc
import os

"""
Are some predictions actually IDENTICAL?
"""

model_name = 'varnet'
p1 = 100
p2 = 25
acc = '10x'
dd2 = f'/home/sharreve/local_scratch/paper/reconstruction/results/{model_name}_parallel_discrete/{p1}p/train_mixed/2ch/{acc}'
dd1 = f'/home/sharreve/local_scratch/paper/reconstruction/results/{model_name}_parallel_discrete/{p2}p/train_mixed/2ch/{acc}'
file_list = os.listdir(dd1)

for i_file in file_list:
    file1 = os.path.join(dd1, i_file)
    file2 = os.path.join(dd2, i_file)
    A = hmisc.load_array(file1, data_key='reconstruction')
    B = hmisc.load_array(file2, data_key='reconstruction')
    fig_obj = hplotc.ListPlot(A - B, cbar=True)
    fig_obj.figure.savefig(os.path.join(DRESULT, f'difference_{i_file}.png'))
    fig_obj = hplotc.ListPlot([A, B], cbar=True)
    fig_obj.figure.savefig(os.path.join(DRESULT, f'comp_{i_file}.png'))
