from objective_configuration.segment7T3T import get_path_dict, DATASET_LIST
import helper.misc as hmisc
import os


"""
We need to report on the size of our datasets...
"""

def _get_total_slices(dfiles):
    file_list = os.listdir(dfiles)
    total_slices = 0
    for i_file in file_list:
        file_path = os.path.join(dfiles, i_file)
        loaded_array = hmisc.load_array(file_path)
        if i_file.endswith('nii.gz'):
            loaded_array = loaded_array.T[:, ::-1, ::-1]
        if loaded_array.ndim == 2:
            loaded_array = loaded_array[None]
        n_slice = loaded_array.shape[0]
        total_slices += n_slice
    return len(file_list), total_slices

from  dataclasses import fields
"""
Metrics for testing dataset
"""

print("Overview test dataset file count and slice count")
for i_dataset in DATASET_LIST:
    path_dict = get_path_dict(i_dataset)
    dimg = path_dict['dimg']
    dlabel = path_dict['dlabel']
    n_files_img, n_slice_img = _get_total_slices(dimg)
    n_files_label, n_slice_label = _get_total_slices(dlabel)
    print('\n\n', i_dataset)
    print('Length test dataset images', n_files_img, n_slice_img)
    print('Length test dataset labels', n_files_label, n_slice_label)

"""
Now the training set sizes...
"""

from nnunet.paths import nnUNet_raw_data
raw_listdir = [os.path.join(nnUNet_raw_data, x) for x in os.listdir(nnUNet_raw_data)]
raw_listdir = [x for x in raw_listdir if os.path.isdir(x)]
for i_nnunet_run in raw_listdir:
    print(os.path.basename(i_nnunet_run))
    json_file = os.path.join(i_nnunet_run, 'dataset.json')
    json_dict = hmisc.load_json(json_file)
    n_test_json = json_dict['numTest']
    n_train_json = json_dict['numTraining']
    print(json_dict['name'])
    print()
    train_img = os.path.join(i_nnunet_run, 'imagesTr')
    train_label = os.path.join(i_nnunet_run, 'labelsTr')
    n_files_train, n_slice_train = _get_total_slices(train_img)
    n_files_train_label, n_slice_train_label = _get_total_slices(train_label)
    print('Number of train files ', n_files_train, n_files_train_label, n_train_json)
    print('Number of train slice ', n_slice_train, n_slice_train_label)
    print("Example training files")
    train_files_sample = [(x['image'], x['label']) for x in json_dict['training']][::int(n_train_json // 10)][:10]
    for ifile in train_files_sample:
        print(ifile)