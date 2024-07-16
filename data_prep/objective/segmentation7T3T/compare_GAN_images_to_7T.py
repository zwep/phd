import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import scipy.stats
import torch
import helper.array_transf as harray
import helper.misc as hmisc
import numpy as np
import helper.plot_class as hplotc
import model.InceptionV3 as inceptionv3
import os
from skimage.metrics import structural_similarity
import ot

"""
We want to select a specific epoch and train stuff on that...
"""


class CompareGAN7T:
    # These are always fixed...
    subdir = 'niftis/cmr3t2cmr7t'
    subdir_mm1 = 'niftis/cmr3t'
    # d7T = '/data/cmr7t3t/cmr7t/RawData_newbatch/nifti_nor_crop'
    d7T = '/data/cmr7t3t/cmr7t/RawData_newbatch/nifti_nor_crop/Image'
    file_list_7T = os.listdir(d7T)

    def __init__(self, dgan, epoch_list, d1p5T, device=None):
        self.n_max = None
        self.dgan = dgan
        self.epoch_list = epoch_list
        self.d1p5T = d1p5T
        self.device = device
        self.model_obj = inceptionv3.InceptionV3()
        self.model_obj.to(device)
        self.ddest_ssim = os.path.join(self.dgan, 'ssim_1p5T_per_epoch.json')
        self.ddest_fid = os.path.join(self.dgan, 'fid_7t_per_epoch.json')

    def load_and_tensorfy(self, ddir, file_name):
        file_path = os.path.join(ddir, file_name)
        loaded_array = hmisc.load_array(file_path)
        if 'nii.gz' in file_name:
            if loaded_array.ndim == 4:
                loaded_array = np.squeeze(loaded_array)
            loaded_array = loaded_array.T[:, ::-1, ::-1]
        else:
            loaded_array = np.squeeze(loaded_array)
        n_loc = loaded_array.shape[0]
        sel_array = loaded_array[n_loc//2]
        sel_array = harray.scale_minmax(sel_array)
        sel_tensor = torch.from_numpy(sel_array).float().to(self.device)[None, None]
        return sel_tensor

    def calculate_wd(self):
        ssim_epoch_dict = {}
        wd_epoch_dict = {}
        for ii, i_epoch_dir in enumerate(self.epoch_list):
            dgan_files = os.path.join(self.dgan, i_epoch_dir, self.subdir)
            file_list_gan = os.listdir(dgan_files)[:self.n_max]
            _ = ssim_epoch_dict.setdefault(i_epoch_dir, [])
            _ = wd_epoch_dict.setdefault(i_epoch_dir, [])
            temp_wd = []
            # temp_ssim = []
            for jj, i_gan_file in enumerate(file_list_gan):
                print(f"N Epoch {ii} / {len(self.epoch_list)}\nN gan {jj} / {len(file_list_gan)}", end='\r')
                sel_gan_tensor = self.load_and_tensorfy(dgan_files, i_gan_file)
                inception_gan = self.model_obj(sel_gan_tensor).cpu()
                sel_temp_wd = []
                for i_7t_file in self.file_list_7T:
                    sel_7t_tensor = self.load_and_tensorfy(self.d7T, i_7t_file)
                    inception_7T = self.model_obj(sel_7t_tensor).cpu()
                    wd_value = ot.sliced_wasserstein_distance(inception_7T, inception_gan)
                    sel_temp_wd.append(float(wd_value))
                temp_wd.append(sel_temp_wd)
            wd_epoch_dict[i_epoch_dir].append(temp_wd)

        avg_fid_dict = harray.aggregate_dict_mean_value(wd_epoch_dict)
        hmisc.store_json(avg_fid_dict, self.ddest_fid)
        return avg_fid_dict

    def calculate_ssim(self):
        """ Now calculate SSIM between 1.5T and GAN synth """
        print('Starting SSIM')

        ssim_epoch_dict = {}
        for ii, i_epoch_dir in enumerate(self.epoch_list):
            dgan_files = os.path.join(dgan, i_epoch_dir, self.subdir)
            dmm1_files = os.path.join(dgan, i_epoch_dir, self.subdir_mm1)
            file_list_gan = os.listdir(dgan_files)[0:self.n_max]
            _ = ssim_epoch_dict.setdefault(i_epoch_dir, [])
            temp_ssim = []
            for jj, i_gan_file in enumerate(file_list_gan):
                print(f"N Epoch {ii} / {len(self.epoch_list)}\nN gan {jj} / {len(file_list_gan)}", end='\r')
                gan_file_path = os.path.join(dgan_files, i_gan_file)
                mm1_file_path = os.path.join(dmm1_files, i_gan_file)
                loaded_gan = np.squeeze(hmisc.load_array(gan_file_path)).T[:, ::-1, ::-1]
                loaded_mm1 = np.squeeze(hmisc.load_array(mm1_file_path)).T[:, ::-1, ::-1]
                n_loc = loaded_gan.shape[0]
                sel_temp_ssim = []
                for sel_loc in range(n_loc):
                    sel_gan = loaded_gan[sel_loc]
                    sel_gan = harray.scale_minmax(sel_gan)
                    sel_mm1 = loaded_mm1[sel_loc]
                    sel_mm1 = harray.scale_minmax(sel_mm1)
                    ssim_value = structural_similarity(sel_mm1, sel_gan)
                    sel_temp_ssim.append(ssim_value)
                temp_ssim.append(sel_temp_ssim)
            ssim_epoch_dict[i_epoch_dir].append(temp_ssim)

        avg_ssim_dict = harray.aggregate_dict_mean_value(ssim_epoch_dict)
        hmisc.store_json(avg_ssim_dict, self.ddest_ssim)
        return avg_ssim_dict

    def visualize_results(self):
        # Somehow.. this import is not detected if put in the preamble
        import matplotlib.pyplot as plt
        """ Visualization and storage of the results... """
        if os.path.isfile(self.ddest_ssim):
            avg_ssim_dict = hmisc.load_json(self.ddest_ssim)
            ssim_label, ssim_value = zip(*sorted([(k, v) for k, v in avg_ssim_dict.items()], key=lambda x: int(x[0][5:])))
            fig, ax = plt.subplots()
            ax.plot(ssim_label, ssim_value)
            ax.tick_params(labelrotation=45)
            fig.savefig(os.path.join(self.dgan, 'ssim_1p5T_per_epoch.png'))

        if os.path.isfile(self.ddest_fid):
            avg_fid_dict = hmisc.load_json(self.ddest_fid)
            fid_label, fid_value = zip(*sorted([(k, v) for k, v in avg_fid_dict.items()], key=lambda x: int(x[0][5:])))
            # Plot it
            fig, ax = plt.subplots()
            ax.plot(fid_label, fid_value)
            ax.tick_params(labelrotation=45)
            fig.savefig(os.path.join(self.dgan, 'fid_7t_per_epoch.png'))

        if os.path.isfile(self.ddest_fid) and os.path.isfile(self.ddest_ssim):
            avg_fid_dict = hmisc.load_json(self.ddest_fid)
            fid_label, fid_value = zip(*sorted([(k, v) for k, v in avg_fid_dict.items()], key=lambda x: int(x[0][5:])))
            avg_ssim_dict = hmisc.load_json(self.ddest_ssim)
            ssim_label, ssim_value = zip(*sorted([(k, v) for k, v in avg_ssim_dict.items()], key=lambda x: int(x[0][5:])))
            # Visualize both..
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2)
            ax[0].plot(fid_label, fid_value, c='red', label='fid')
            ax[0].plot(ssim_label, ssim_value, c='blue', label='ssim')
            ax[0].legend()
            ax[0].tick_params(labelrotation=45)
            ax[1].scatter(fid_value, ssim_value)
            ax[1].set_xlabel('fid (GAN, 7T) per epoch')
            ax[1].set_ylabel('sim (GAN, 1.5T) per epoch')
            fig.savefig(os.path.join(self.dgan, 'ssim_and_fid_7t_per_epoch.png'))



# python -m pytorch_fid /data/cmr7t3t/mms1_synthesis_220725/seven_mms1A_cut_220720/test_100/niftis/cmr3t2cmr7t /data/cmr7t3t/cmr7t/RawData_newbatch/data_nifti_ED_ES_crop
if __name__ == "__main__":
    # dgan = f'/data/cmr7t3t/mms1_synthesis_220725/seven_mms1{vendor}_cut_220720'
    vendor_list = ['A', 'B']
    for vendor in vendor_list:
        d1p5T = f'/data/cmr7t3t/mms1/all_phases_mid/Vendor_{vendor}'
        dgan_base = '/data/cmr7t3t/mms1_synthesis_220908'
        sub_dir_list = [f'new_seven_mms1{vendor}_cut_NCE8_GAN2_220907', f'new_seven_mms1{vendor}_cycleGAN_lambda_A30B30_220907']
        for sub_dir in sub_dir_list:
            dgan = os.path.join(dgan_base, sub_dir)
            epoch_list = os.listdir(dgan)
            epoch_list = [x for x in epoch_list if os.path.isdir(os.path.join(dgan, x))]
            index_gpu = 0
            device = torch.device("cuda:{}".format(str(index_gpu)) if torch.cuda.is_available() else "cpu")
            # Create the compare object
            compare_obj = CompareGAN7T(dgan=dgan, epoch_list=epoch_list, device=device, d1p5T=d1p5T)
            avg_fid_dict = compare_obj.calculate_wd()
            avg_ssim_dict = compare_obj.calculate_ssim()
            # STore the results
            compare_obj.visualize_results()
            # compare_obj.visualize_results(avg_ssim_dict=avg_ssim_dict, avg_fid_dict=avg_fid_dict)
            # compare_obj.visualize_results()
