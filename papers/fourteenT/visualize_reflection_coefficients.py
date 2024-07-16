import numpy as np
import pandas as pd
import os
import re
import helper.plot_class as hplotc
from objective_configuration.fourteenT import DDATA

"""
Code for Reading and Plotting S-Parameter Data from Excel Files

This code reads in a list of Excel files with S-parameter data, converts the data into
complex S-matrix objects, and plots the absolute values of the matrices using the
helper.plot_class module. 
The Excel files must be in a specific format, with columns
labeled "Sij(i,j) [real]" and "Sij(i,j) [imag]" for each element of the S-matrix. The
code reads in all Excel files in the specified directory and generates a plot with
subplots for each file.
The subplot titles are derived from the filenames.
"""


def convert_xlsx_to_array (xls_obj, n_coils):
    S_matrix = np.zeros((n_coils, n_coils), dtype=complex)
    column_names = xls_obj.columns
    re_index = re.compile('Sij\(([0-9]+,[0-9]+)\)\s\[(\w+)\]')
    for i_col in column_names:
        s_ij = xls_obj[i_col].values[0]
        re_find = re_index.findall(i_col)
        if re_find:
            ij_str, cpx_type = re_find[0]
            i, j = map(int, ij_str.split(","))
            if cpx_type == 'real':
                S_matrix[i-1, j-1] += s_ij
            else:
                S_matrix[i - 1, j - 1] += 1j * s_ij

    return S_matrix

file_list = os.listdir(DDATA)
xlsx_files = [x for x in file_list if x.endswith('xlsx')]
n_xlxs_files = len(xlsx_files)

sub_title_list = []
S_matrix_list = []
for i_xlsx in xlsx_files:
    n_channel = int(re.findall('([0-9]+) Channel', i_xlsx)[0])
    file_path = os.path.join(DDATA, i_xlsx)
    title = re.sub(' - Sij.xlsx', '', i_xlsx)
    # Read the xls..
    xls_obj = pd.read_excel(file_path)
    S_matrix = convert_xlsx_to_array(xls_obj, n_channel)
    S_matrix_list.append(S_matrix)
    sub_title_list.append(title)

hplotc.ListPlot([S_matrix_list], augm='np.abs', subtitle=[sub_title_list], sub_col_row=(3, 2), hspace=0.2, ax_off=True, cbar=True)