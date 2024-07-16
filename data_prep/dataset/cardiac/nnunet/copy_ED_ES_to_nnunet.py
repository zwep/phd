import shutil
import helper.misc as hmisc
import os

"""
We have made a selection of ED and ES slices and would like to copy them to an nnunet folder where it can be preprocessed

This means that we need to do a name change... (append a _0000)
create a new datajson
prepare data set

then run nnunet models on this dataset, store it in the folder /Results in cmrt7T
"""

dsource = '/data/cmr7t3t/cmr7t/Image_ED_ES'
dtarget = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task999_7T/imagesTs'


for source_file in os.listdir(dsource):
    base_name = hmisc.get_base_name(source_file)
    ext = hmisc.get_ext(source_file)
    target_file = base_name + "_0000" + ext
    source_path = os.path.join(dsource, source_file)
    target_path = os.path.join(dtarget, target_file)
    print(f'Copying {source_path} to {target_path}')
    shutil.copy(source_path, target_path)
