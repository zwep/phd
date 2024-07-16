import os
import helper.misc as hmisc
import sys

"""
I cant create a separate file for every Task combination
Currently I want to combine them in a trivial way: simply by merging two folders"""


raw_data_path = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/'

"""Two datasets combined: """

#       Task602_MM1_A_Biasfield_MM1_A:
# Task601_Biasfield_MM1_A
# Task501_MM1_A
ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task602_MM1_A_Biasfield_MM1_A'
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=())
os.system(f"cp -r {raw_data_path}Task601_Biasfield_MM1_A/* {ddest}")
os.system(f"cp -r {raw_data_path}Task501_MM1_A/* {ddest}")
# datasetjsonoshfklsjeijshgf
# nnUNet_plan_and_preprocess -t 602 --verify_dataset_integrity

#       Task604_MM1_B_Biasfield_MM1_B:
# Task603_Biasfield_MM1_B
# Task502_MM1_B
ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task604_MM1_B_Biasfield_MM1_B'
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=())
os.system(f"cp -r {raw_data_path}Task603_Biasfield_MM1_B/* {ddest}")
os.system(f"cp -r {raw_data_path}Task502_MM1_B/* {ddest}")
# create dataset json..
# nnUNet_plan_and_preprocess -t 604 --verify_dataset_integrity


"""Three datasets combined: """

#       Task651_MM1_A_ACDC_Biasfield_MM1_A:
# Task602_MM1_A_Biasfield_MM1_A
# Task511_ACDC
ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task651_MM1_A_ACDC_Biasfield_MM1_A'
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=())
os.system(f"cp -r {raw_data_path}Task602_MM1_A_Biasfield_MM1_A/* {ddest}")
os.system(f"cp -r {raw_data_path}Task511_ACDC/* {ddest}")
# create datasetjson..
# nnUNet_plan_and_preprocess -t 651 --verify_dataset_integrity

#       Task653_MM1_B_ACDC_Biasfield_MM1_B
# Task604_MM1_B_Biasfield_MM1_B
# Task511_ACDC
ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task653_MM1_B_ACDC_Biasfield_MM1_B'
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=())
os.system(f"cp -r {raw_data_path}Task604_MM1_B_Biasfield_MM1_B/* {ddest}")
os.system(f"cp -r {raw_data_path}Task511_ACDC/* {ddest}")
# create datasetjson..
# nnUNet_plan_and_preprocess -t 653 --verify_dataset_integrity

#

#Task630
ddest = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task630_MM1_A_MM1_B_ACDC'
hmisc.create_datagen_dir(ddest, type_list=('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'), data_list=())
os.system(f"cp -r {raw_data_path}Task501_MM1_A/* {ddest}")
os.system(f"cp -r {raw_data_path}Task502_MM1_B/* {ddest}")
os.system(f"cp -r {raw_data_path}Task511_ACDC/* {ddest}")