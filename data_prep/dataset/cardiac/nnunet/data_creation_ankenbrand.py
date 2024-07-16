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
ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task513_Kaggle'
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=())

score_df = pandas.read_csv(dcsv, sep='\t')
sel_row = score_df['score'] <= 0.05
sel_score_df = score_df.loc[sel_row].sort_values('score').reset_index(drop=True)

max_train_examples = 125
max_test_examples = int(max_train_examples * (3/7))

counter_files = 0
counter_row = 0
first_time_switch = True
while counter_row < 10000:
    if counter_files <= max_train_examples:
        print(f'derp {counter_row} train', end='\r')
        ddest_img = os.path.join(ddest, 'imagesTr')
        ddest_label = os.path.join(ddest, 'labelsTr')
    elif counter_files <= max_train_examples + max_test_examples:
        if first_time_switch:
            # This makes sure we take totally different images...
            counter_row += 200
            first_time_switch = False
        print(f'derp {counter_row} test', end='\r')
        ddest_img = os.path.join(ddest, 'imagesTs')
        ddest_label = os.path.join(ddest, 'labelsTs')
    else:
        break
    #
    irow = sel_score_df.iloc[counter_row]
    counter_row += 1
    base_name = hmisc.get_base_name(irow['file'])
    slice_nr = int(re.findall('slice([0-9]{3})', base_name)[0])
    # Filter on middle slices...
    if slice_nr in [3, 4, 5]:
        counter_files += 1
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

