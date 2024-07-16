import os
import re
import helper.misc as hmisc
import pandas
import nibabel
import numpy as np
"""
The work by Ankenbrand contains good quality images..
Lets extract them.. and run inference on it.

"""

dcsv = '/data/cmr7t3t/ankenbrand_data/data/image_list_filtered_score.tsv'
ddata = '/data/cmr7t3t/ankenbrand_data/data'
ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task998_ankenbrand'
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=())
ddest_img = os.path.join(ddest, 'imagesTs')
ddest_label = os.path.join(ddest, 'labelsTs')

score_df = pandas.read_csv(dcsv, sep='\t')
sel_row = score_df['score'] <= 0.05
sel_score_df = score_df.loc[sel_row].sort_values('score').reset_index(drop=True)
id_without_rot = sorted(list(set([x for x in list(sel_score_df['Id']) if '_' not in x])), key=lambda x: int(x))

for ii, i_id in enumerate(id_without_rot):
    # Get the file list of a specific id
    temp_id_df = sel_score_df[sel_score_df['Id'] == i_id].reset_index(drop=True)
    temp_id_df['slice_nr'] = temp_id_df['file'].apply(lambda x: int(re.findall('slice([0-9]{3})', x)[0]))
    # Calculate max/min slice
    slice_number_list = [int(re.findall('slice([0-9]{3})', x)[0]) for x in list(temp_id_df['file'])]
    min_slice = min(slice_number_list)
    max_slice = max(slice_number_list)
    for i_slice in range(max_slice//2-1, max_slice//2+1):
        irow = temp_id_df[temp_id_df['slice_nr'] == i_slice].sample()
        # This is so that we can select the row itself and get a Series back
        irow = irow.reset_index(drop=True).iloc[0]
        base_name = hmisc.get_base_name(irow['file'])
        # source
        dsource_img = os.path.join(ddata, irow['file'])
        dsource_label = re.sub('images', 'masks', dsource_img)
        # target
        dtarget_img = os.path.join(ddest_img, base_name + '_0000' + '.nii.gz')
        dtarget_label = os.path.join(ddest_label, base_name + '.nii.gz')
        img_array = hmisc.load_array(dsource_img, convert2gray=True)[:, :, 0:1]
        label_array = hmisc.load_array(dsource_label)[:, :, 0:1]
        # Change labels
        label_index_1 = label_array == 1
        label_index_3 = label_array == 3
        label_array[label_index_1] = 3
        label_array[label_index_3] = 1
        nibabel_obj = nibabel.Nifti1Image(img_array, np.eye(4))
        nibabel.save(nibabel_obj, dtarget_img)
        nibabel_obj = nibabel.Nifti1Image(label_array, np.eye(4))
        nibabel.save(nibabel_obj, dtarget_label)

