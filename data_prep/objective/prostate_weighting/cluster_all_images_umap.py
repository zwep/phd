import os
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
from skimage.util import img_as_ubyte, img_as_uint
import os
import helper.misc as hmisc
import helper.metric as hmetric
import helper.array_transf as harray
import pandas as pd
import skimage.transform as sktransform
import multiprocessing as mp
"""
First get an ssim matrix... or something
a distance matrix, affinity matrix what ever
"""

class CalculateDistanceGraph:
    def __init__(self, dict_dir, dest_dir, ext='nii.gz'):
        self.dir_labels, self.dir_paths = zip(*[[k, v] for k, v in dict_dir.items()])
        self.dest_dir = dest_dir
        self.all_files = []
        self.all_labels = []
        for k, v in dict_dir.items():
            file_list = [os.path.join(v, x) for x in os.listdir(v) if x.endswith(ext)]
            temp_n_files = len(file_list)
            file_label = [k] * temp_n_files
            self.all_files.extend(file_list)
            self.all_labels.extend(file_label)
        self.n_files = len(self.all_files)
    @staticmethod
    def prep_file(x):
        if x.ndim == 4:
            x = np.squeeze(x)
        x = x.T[:, ::-1, ::-1]
        n_slice = x.shape[0]
        # Getting only one slice.... understandable..
        # This is to limit clusters
        # But then I should do the same for the bias field synthetic images..
        sel_slice = n_slice // 2
        x = x[sel_slice]
        return x
    def get_ssim_matrix(self):
        for i_file in range(self.n_files):
            print(f"{i_file} / {self.n_files}", end='\r')
            sel_file = self.all_files[i_file]
            sel_label = self.all_labels[i_file]
            i_base_name = hmisc.get_base_name(sel_file)
            temp_dict = {(sel_label, i_base_name): np.zeros(self.n_files)}
            i_x = hmisc.load_array(sel_file)
            i_x = self.prep_file(i_x)
            i_x = harray.scale_minmax(i_x)
            # Check what the shape is...
            print(f"{i_base_name} - shape ", i_x.shape)
            # print('\t ', temp_dict)
            for j_file in range(i_file, self.n_files):
                counter_file = self.all_files[j_file]
                j_x = hmisc.load_array(counter_file)
                j_x = self.prep_file(j_x)
                j_x = harray.scale_minmax(j_x)
                ssim_value = structural_similarity(img_as_ubyte(i_x), img_as_ubyte(j_x))
                temp_dict[(sel_label, i_base_name)][j_file] = ssim_value
            temp_df = pd.DataFrame.from_dict(temp_dict, orient='index')
            temp_df.to_csv(os.path.join(self.dest_dir, 'ssim_matrix.csv'), mode='a', index=True, header=False)
    def get_ssim_matrix_mp(self):
        N = mp.cpu_count()
        with mp.Pool(processes=N//4) as p:
            p.map(self.calc_single_map, list(range(self.n_files)))

    def calc_single_map(self, i_file):
        sel_file = self.all_files[i_file]
        sel_label = self.all_labels[i_file]
        i_base_name = hmisc.get_base_name(sel_file)
        temp_dict = {(sel_label, i_base_name): np.zeros(self.n_files)}
        i_x = hmisc.load_array(sel_file)
        i_x = self.prep_file(i_x)
        i_x = harray.scale_minmax(i_x)
        for j_file in range(i_file, self.n_files):
            counter_file = self.all_files[j_file]
            j_x = hmisc.load_array(counter_file)
            j_x = self.prep_file(j_x)
            j_x = harray.scale_minmax(j_x)
            # ssim_value = structural_similarity(img_as_ubyte(i_x), img_as_ubyte(j_x))
            contrast_value = hmetric.get_contrast_ssim(i_x, j_x)
            temp_dict[(sel_label, i_base_name)][j_file] = contrast_value#ssim_value
        temp_df = pd.DataFrame.from_dict(temp_dict, orient='index')
        temp_df.to_csv(os.path.join(self.dest_dir, 'contrast_matrix.csv'), mode='a', index=True, header=False)


if __name__ == "__main__":
    dir_list = {'MM1': '/data/cmr7t3t/mms1/all_phases_mid/Vendor_A/Image_single_slice',
                'ACDC': '/data/cmr7t3t/acdc/acdc_processed/Image',
                '7T': '/data/cmr7t3t/cmr7t/Image_ED_ES',
                '7T_biasf': '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task611_Biasfield_ACDC/imagesTr',
                '7T_synth': '/data/cmr7t3t/results/ACDC_220121/seven2three_acdc_cut_NCE4_GAN2_np128_fe_211208/test_100/niftis/cmr3t2cmr7t'}
    dist_obj = CalculateDistanceGraph(dict_dir=dir_list, dest_dir='/data/seb')
    dist_obj.get_ssim_matrix_mp()
    # dist_obj.get_ssim_matrix()
