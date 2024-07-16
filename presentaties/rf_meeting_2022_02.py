import nibabel
import helper.plot_class as hplotc
import helper.misc as hmisc
import os
import numpy as np

# ddata = '/home/bugger/Documents/presentaties/RF_meetings/RF_Meeting_20220222/nnunet_3T_ACDC_examples'
# ddata = '/home/bugger/Documents/presentaties/RF_meetings/RF_Meeting_20220222/7T_biasfield_examples'
# ddata = '/home/bugger/Documents/presentaties/RF_meetings/RF_Meeting_20220222/7T_synth'
ddata = '/home/bugger/Documents/presentaties/RF_meetings/RF_Meeting_20220222/7T_nnunet_examples'

dimg = os.path.join(ddata, 'image')
dlabel = os.path.join(ddata, 'label')
dpred = os.path.join(ddata, 'pred')

img_list = []
for i_file in os.listdir(dimg):
    img_file = os.path.join(dimg, i_file)
    label_file = os.path.join(dlabel, i_file)
    pred_file = os.path.join(dpred, i_file)
    temp_list = []
    for sel_file in [img_file, label_file, pred_file]:
        temp_img = hmisc.load_array(sel_file)[:, :]
        temp_list.append(temp_img)

    img_list.append(temp_list)

fig_handle = hplotc.ListPlot(img_list, ax_off=True, wspace=0, hspace=0, figsize=(10, 20))
fig_handle.figure.savefig(os.path.join(ddata, 'example.png'), bbox_inches='tight')
hplotc.close_all()
