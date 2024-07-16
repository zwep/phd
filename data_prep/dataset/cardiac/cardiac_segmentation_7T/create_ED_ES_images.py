import nibabel
import os
import helper.plot_class as hplotc
import numpy as np
import pandas as pd

"""
Here we only create the ED and ES images from the overview_ed_es_slice.csv since Yasmina/Sina created their ED/ES labels directly

With a small change we can do the same for the labels..
"""

dimage = '/data/cmr7t3t/cmr7t/Image_selection'
#dimage = '/data/cmr7t3t/cmr7t/Label'
dED_ES = '/data/cmr7t3t/cmr7t/overview_ed_es_slice.csv'
ddest_image = '/data/cmr7t3t/cmr7t/Image_ED_ES'
# ddest_image = '/data/cmr7t3t/cmr7t/Label_ED_ES'

csv_obj = pd.read_csv(dED_ES)

for ii, i_row in csv_obj.iterrows():
    subject_name = i_row['Subject name']
    ED_index = i_row['ED slice number']
    ES_index = i_row['ES slice number']
    if ED_index is not None:
        if not np.isnan(ED_index):
            ED_index = int(ED_index)
            ES_index = int(ES_index)
            image_file = os.path.join(dimage, subject_name)
            image_nibabel = nibabel.load(image_file)
            image_array = image_nibabel.get_fdata()
            print("\nSubject name ", subject_name)
            print("\tED index ", ED_index)
            print("\tES index ", ES_index)
            print("\timage shape ", image_array.shape)
            image_ED = image_array[:, :, ED_index]
            image_ES = image_array[:, :, ES_index]

            temp_dict = {'ES': (image_nibabel.affine, image_ES), 'ED': (image_nibabel.affine, image_ED)}

            for subdir, container in temp_dict.items():
                if 'ES' in subdir:
                    dest_path = os.path.join(ddest_image, 'ES_' + subject_name)
                else:
                    dest_path = os.path.join(ddest_image, 'ED_' + subject_name)

                affine_struct = container[0]
                i_image = container[1]
                nibabel_obj = nibabel.Nifti1Image(i_image[:, :, None], affine_struct)
                print("\t Target path ", dest_path)
                try:
                    nibabel.save(nibabel_obj, dest_path)
                except:
                    print("No ownership")
