import nibabel
import os

ddata = '/data/cmr7t3t/acdc/acdc_processed/Image'
ddata = '/data/cmr7t3t/acdc/acdc_processed/ImageTest'

s = 0
for i_file in os.listdir(ddata):
    sel_file = os.path.join(ddata, i_file)
    temp_img = nibabel.load(sel_file).get_fdata()
    s += temp_img.shape[-1]
