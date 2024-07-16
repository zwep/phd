"""
It seems as if ... the process does not respect the file names...

"""

ddata = '/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input'

import os
import helper.misc as hmisc
tot_slice = 0
file_list = os.listdir(ddata)
n = len(file_list)
for i_file in file_list[:int(n*0.25)]:
    file_path = os.path.join(ddata, i_file)
    A = hmisc.load_array(file_path, data_key='kspace')
    print(A.shape)
    n_slice = A.shape[0]
    tot_slice += n_slice


print(tot_slice)


"""
I found the following files...
"""



z =['/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_02052021_1026149_10_3_p2ch_radialV4.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_06022021_1231006_10_3_p2ch_radialV4.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_02052021_1451162_15_3_4ch_radialV4.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_06032021_1107269_13_3_sa_radialV4_02.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_03022021_1651308_10_3_p2ch_radialV4.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_03022021_1651308_10_3_p2ch_radialV4.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_07032021_1230000_6_3_transradialfastV4.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_02052021_1143280_17_3_sa_radialV4_01.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_06032021_1327043_12_3_sa_radialV4_00.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_02052021_0939191_14_3_sa_radialV4_02.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_02052021_1451162_15_3_4ch_radialV4.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_13022021_1212010_9_3_transradialfastV4.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_02052021_1245354_17_3_4ch_radialV4.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_02052021_1128409_8_3_transradialfastV4.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_06022021_1151300_16_3_4ch_radialV4.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_03022021_1651308_10_3_p2ch_radialV4.h5'
,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_03022021_1656244_12_3_sa_radialV4_00.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_03022021_1656244_12_3_sa_radialV4_01.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_02052021_1143280_17_3_sa_radialV4_01.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_13022021_1212010_9_3_transradialfastV4.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_02052021_1337561_12_3_sa_radialV4_03.h5'

,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_02052021_1444045_13_3_sa_radialV4_04.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_02052021_1245354_17_3_4ch_radialV4.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_06032021_1327043_12_3_sa_radialV4_00.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_02052021_0939191_14_3_sa_radialV4_02.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_03022021_1656244_12_3_sa_radialV4_00.h5'


,'/home/sharreve/local_scratch/mri_data/cardiac_full_radial/mixed/train/input/v9_13022021_1227048_16_3_sa_radialV4_04.h5']

dd25 = '/home/bugger/Documents/data/recon/train_25.lst'
dd50 = '/home/bugger/Documents/data/recon/train_50.lst'
dd75 = '/home/bugger/Documents/data/recon/train_75.lst'
dd100 = '/home/bugger/Documents/data/recon/train_100.lst'

with open(dd25, 'r') as f:
    list_25 = [x.strip() for x in f.readlines()]

with open(dd50, 'r') as f:
    list_50 = [x.strip() for x in f.readlines()]

with open(dd75, 'r') as f:
    list_75 = [x.strip() for x in f.readlines()]

with open(dd100, 'r') as f:
    list_100 = [x.strip() for x in f.readlines()]

import os
z_basename = [os.path.basename(x) for x in z]

s25 = 0
s50 = 0
s75 = 0
s100 = 0
for i_printed in z_basename:
    if i_printed in list_25:
        s25 += 1
    if i_printed in list_50:
        s50 += 1
    if i_printed in list_75:
        s75 += 1
    if i_printed in list_100:
        s100 += 1

print(s25, s50, s75, s100)
