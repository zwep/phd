import re
import helper.plot_class as hplotc
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import os

"""
Construct a generator for all the data from Daan Reesink
"""
"""
# THIs order is for file 7TMRI010
pydicom_array[-1] = pydicom_array[-1][[0, 9, 5, 14, 1, 10, 6, 15, 2, 11, 7, 16, 3, 12, 8, 17, 4, 13]]

"""


class GeneratorPatientData:
    # This can be used to eventually spit out the correct order of the images as well...
    image_order_patient = {"7TMRI010": [0, 9, 5, 14, 1, 10, 6, 15, 2, 11, 7, 16, 3, 12, 8, 17, 4, 13]}

    def __init__(self, ddata):
        self.ddata = ddata
        self.file_list = self.get_file_list()
        self.patient_id_list = self.get_patient_id_list()
        self.patient_file_list = self.get_file_mask_slice_list()

    def get_file_list(self):
        t2w_file_list = []
        for d, _, f in os.walk(self.ddata):
            filter_f = [x for x in f if 't2w' in x.lower() and x.endswith('dcm')]
            if len(filter_f):
                for i_file in filter_f:
                    temp_file = os.path.join(d, i_file)
                    t2w_file_list.append(temp_file)

        return t2w_file_list

    def get_patient_id_list(self):
        patient_id_pattern = re.compile('(7TMRI[0-9]{3})')
        patient_id_list = list(set([patient_id_pattern.findall(x)[0] for x in self.file_list if patient_id_pattern.findall(x)]))
        patient_id_list = sorted(patient_id_list)
        return patient_id_list

    def get_file_mask_slice_list(self):
        file_mask_slice_list = []
        for i_patient in self.patient_id_list:
            multiple_files = False
            filter_t2w_file_list = [x for x in self.file_list if i_patient in x]
            if len(filter_t2w_file_list) > 1:
                multiple_files = True
                n_slice = len(filter_t2w_file_list)
            else:
                sel_file = filter_t2w_file_list[0]
                temp_array = pydicom.read_file(sel_file).pixel_array
                n_slice = temp_array.shape[0]

            for i_file in filter_t2w_file_list:
                base_name, _ = os.path.splitext(i_file)
                sel_mask_file = base_name + "_mask.npy"
                for i_slice in range(n_slice):
                    if multiple_files:
                        i_slice = None

                    file_mask_slice_list.append({'file_name': i_file, 'mask_name': sel_mask_file, 'slice': i_slice})
        return file_mask_slice_list

    def __getitem__(self, item):
        return self.patient_file_list[item]

    def __len__(self):
        return len(self.patient_file_list)


if __name__ == "__main__":
    # Check the standard deviation outside the body
    ddata = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter'
    ddata = '/home/bugger/Documents/data/7T/prostate/DICOM'
    generator_files = GeneratorPatientData(ddata=ddata)
    generator_files.file_list
    all_std_values = []
    for container in generator_files:
        sel_slice = container['slice']
        temp_array = np.squeeze(pydicom.read_file(container['file_name']).pixel_array[sel_slice])
        mask_file = container['mask_name']
        if os.path.isdir(mask_file):
            temp_mask = np.squeeze(np.load(container['mask_name'])[sel_slice])
        else:
            import helper.array_transf as harray
            temp_mask = harray.get_treshold_label_mask(temp_array, class_treshold=0.05, treshold_value=0.5 * np.mean(temp_array))

        print('SNR', np.mean(temp_array[temp_mask == 1]) / np.std(temp_array[temp_mask == 0]))

        temp_std = 0.5 * np.mean((temp_array[temp_mask == 0]) ** 2)
        # plt.hist((temp_array / temp_std).ravel(), bins=256)
        all_std_values.append(temp_std / np.max(temp_array))

    import matplotlib.pyplot as plt
    plt.hist(all_std_values, bins=128)
    np.mean(all_std_values)

    # # #
    import h5py
    ddata_h5 = '/home/bugger/Documents/data/3T/prostate/prostate_weighting/test/target/40_MR.h5'
    h5_obj = h5py.File(ddata_h5, 'r')
    data_array = h5_obj['data'][50]
    mask_array = harray.get_treshold_label_mask(data_array)
    print('SNR 1.5T image', np.mean(data_array[mask_array == 1]) / np.std(data_array[mask_array == 1]))