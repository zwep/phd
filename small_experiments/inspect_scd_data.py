
# Hier even testen iets van SCD data
import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np

dir_segment = '/media/bugger/MyBook/data/scd/scd_manualcontours/SCD_ManualContours/SC-HF-I-01/contours-manual/IRCCI-expert'
sel_file = os.listdir(dir_segment)[0]
sel_file = os.path.join(dir_segment, sel_file)
with open(sel_file, 'r') as f:
    A = f.readlines()

A_coord = [list(map(float, x.strip().split())) for x in A]
A_coord = np.array(A_coord)
plt.scatter(A_coord[:, 0], A_coord[:, 1])

dir_patient_1 = '/media/bugger/MyBook/data/scd/SCD_DeidentifiedImages/SCD0000101'
for i_dir in os.listdir(dir_patient_1)[1:2]:
    dir_time = f'/media/bugger/MyBook/data/scd/SCD_DeidentifiedImages/SCD0000101/{i_dir}'
    A_time = []
    for i_file in os.listdir(dir_time):
        if i_file.endswith('dcm'):
            sel_file = os.path.join(dir_time, i_file)
            A = pydicom.read_file(sel_file).pixel_array
            A_time.append(A)

    import helper.plot_class as hplotc
    hplotc.ListPlot(np.array(A_time)[0])
    hplotc.SlidingPlot(np.array(A_time))