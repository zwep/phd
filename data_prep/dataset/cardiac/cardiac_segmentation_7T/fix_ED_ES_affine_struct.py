import os
import nibabel
import numpy as np
import re

dimage_all = '/data/cmr7t3t/cmr7t/Image_all'
dimage = '/data/cmr7t3t/cmr7t/Image_ED_ES'
dlabel = '/data/cmr7t3t/cmr7t/Label_ED_ES'

used_structs = []
for i_file in os.listdir(dlabel):
    label_path = os.path.join(dlabel, i_file)
    label_nibabel_obj = nibabel.load(label_path)
    label_struct = label_nibabel_obj.affine
    nibabel_obj = nibabel.Nifti1Image(label_nibabel_obj.get_fdata(), np.eye(4))
    nibabel.save(nibabel_obj, label_path)

for i_file in os.listdir(dimage):
    img_path = os.path.join(dimage, i_file)
    img_nibabel_obj = nibabel.load(img_path)
    img_struct = img_nibabel_obj.affine
    nibabel_obj = nibabel.Nifti1Image(img_nibabel_obj.get_fdata(), np.eye(4))
    nibabel.save(nibabel_obj, img_path)


for i_file in os.listdir(dimage_all):
    img_path = os.path.join(dimage_all, i_file)
    img_nibabel_obj = nibabel.load(img_path)
    img_struct = img_nibabel_obj.affine
    nibabel_obj = nibabel.Nifti1Image(img_nibabel_obj.get_fdata(), np.eye(4))
    nibabel.save(nibabel_obj, img_path)