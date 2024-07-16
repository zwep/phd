"""
here we need to split the following directories into train/test/val...

/data/seb/radial_unprocessed/cartesian_cardiac_cine/h5_data/{2ch, 4ch, trans, sa}

"""

import helper.misc as hmisc

ddir = '/data/seb/radial_unprocessed/cartesian_cardiac_cine/h5_data/4ch'
dtarget = '/tank/data/seb/radial_unprocessed/cartesian_cardiac_cine/split_data/4ch'
hmisc.create_and_copy_data_split(source_dir=ddir, target_dir=dtarget, sel_target_type='input')

import helper.misc as hmisc

ddir = '/data/seb/radial_unprocessed/cartesian_cardiac_cine/h5_data/2ch'
dtarget = '/data/seb/radial_unprocessed/cartesian_cardiac_cine/split_data/2ch'
hmisc.create_and_copy_data_split(source_dir=ddir, target_dir=dtarget, sel_target_type='input')

import helper.misc as hmisc

ddir = '/data/seb/radial_unprocessed/cartesian_cardiac_cine/h5_data/sa'
dtarget = '/data/seb/radial_unprocessed/cartesian_cardiac_cine/split_data/sa'
hmisc.create_and_copy_data_split(source_dir=ddir, target_dir=dtarget, sel_target_type='input')


import helper.misc as hmisc

ddir = '/data/seb/radial_unprocessed/cartesian_cardiac_cine/h5_data/transverse'
dtarget = '/tank/data/seb/radial_unprocessed/cartesian_cardiac_cine/split_data/transverse'
hmisc.create_and_copy_data_split(source_dir=ddir, target_dir=dtarget, sel_target_type='input')
