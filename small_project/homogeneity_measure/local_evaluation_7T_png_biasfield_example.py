from PIL import Image
from skimage.util import img_as_ubyte, img_as_uint
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
import small_project.homogeneity_measure.metric_implementations as hhomog

"""
instead of using the signal intensity we use the biasfield and fit some polynomial to it
"""

ddata = f"/home/bugger/Documents/paper/inhomogeneity removal/result_models/patient_data"
glcm_metric = 'homogeneity'

patient_contrast = {}
for i_dir in os.listdir(ddata):
    sel_ddata = os.path.join(ddata, i_dir)
    if os.path.isdir(sel_ddata):
        dest_mask_file = os.path.join(sel_ddata, 'mask_closeup.npy')
        loaded_dict = {}
        for i_file in os.listdir(sel_ddata):
            if 'closeup' in i_file and 'mask' not in i_file:
                file_path = os.path.join(sel_ddata, i_file)
                loaded_array = hmisc.load_array(file_path)
                loaded_dict[i_file] = loaded_array

        # Create a mask.
        if os.path.isfile(dest_mask_file):
            mask_npy = np.load(dest_mask_file)
        else:
            pillow_obj = Image.open(os.path.join(sel_ddata, 'uncorrected_closeup.png'))
            loaded_array = np.array(pillow_obj)
            mask_obj = hplotc.MaskCreator(loaded_dict['single_homog_closeup.png'])
            mask_npy = mask_obj.mask
            np.save(dest_mask_file, mask_npy)

        patient_contrast.setdefault(i_dir, {})

        for file_name, file_array in loaded_dict.items():
            file_array = harray.scale_minmax(file_array)
            file_array = img_as_ubyte(file_array)
            a, b = np.histogram((file_array[mask_npy == 1]).ravel(), bins=256)
            most_occuring_value = b[np.argmax(a)]
            file_array = (file_array - most_occuring_value) + 256 // 2
            file_array[file_array < 0] = 0
            file_array[file_array > 255] = 255
            patient_contrast[i_dir].setdefault(file_name, [])
            """
            The variance of the fully or a partially segmented
            image, which is supposed to decrease after the IIH
            correction. 
            """
            std_value = np.std(file_array[mask_npy==1])
            patient_contrast[i_dir][file_name].append(std_value)
            """
            It can be shown that this quantity overcomes the lim-
            itation of the image variance. But the cv alone only
            characterizes the within-class scatter
            """
            cov_variation = np.std(file_array[mask_npy == 1]) / np.mean(file_array[mask_npy == 1])
            patient_contrast[i_dir][file_name].append(cov_variation)
            """
            The coefficient of joint variations between two classes: takes into account extra-class variation
            Lets do the prostate as one class... and the not-prostate as the other...???
            """
            nominator = (np.std(file_array[mask_npy == 1]) + np.std(file_array[mask_npy != 1]))
            denominator = np.abs(np.mean(file_array[mask_npy == 1]) - np.mean(file_array[mask_npy != 1]))
            cov_joint_variation = nominator / denominator
            patient_contrast[i_dir][file_name].append(cov_joint_variation)

            # Stability of the biasfield... what if we do it multiple times??

hmisc.print_dict(patient_contrast)


json_ser_obj = json.dumps(patient_contrast)
with open(os.path.join(ddata, f'overview_{glcm_metric}_biasfield_closeup.json'), 'w') as f:
    f.write(json_ser_obj)

# Now a visualization of this....
with open(os.path.join(ddata, f'overview_{glcm_metric}_biasfield_closeup.json'), 'r') as f:
    text_obj = f.read()
    patient_contrast = json.loads(text_obj)

biasfield_contrast = []
direct_contrast = []
uncor_contrast = []
for k, v in patient_contrast.items():
    for ik, iv in v.items():
        if 'biasf' in ik:
            biasfield_contrast.append(iv)
        elif 'homog' in ik:
            direct_contrast.append(iv)
        elif 'uncor' in ik:
            uncor_contrast.append(iv)

plt.figure()
plt.plot(uncor_contrast)
plt.plot(biasfield_contrast)
plt.plot(direct_contrast)

fig, ax = plt.subplots()
plt.bar(np.arange(len(biasfield_contrast)), direct_contrast, color='b', width=0.2, label='direct')
plt.bar(0.2 + np.arange(len(biasfield_contrast)), biasfield_contrast, color='r', width=0.2, label='biasfield')
plt.bar(0.4 + np.arange(len(biasfield_contrast)), uncor_contrast, color='k', width=0.2, label='uncor')
plt.legend()
