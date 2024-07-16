import os
os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10" # export NUMEXPR_NUM_THREADS=6
from objective_configuration.reconstruction import DDATA_spoked, DDATA_sin, DDATA
import itertools
import time
import sigpy
import sigpy.mri
import numpy as np
import helper.plot_class as hplotc
import helper.reconstruction as hrecon
import objective_helper.reconstruction as hobjrecon
import helper.misc as hmisc
import helper.array_transf as harray

"""
We need to convert the datasets to radial spokes... using the sin-files retrieve with a previous script.

I also need to check if the backprojection method works when combining coils, complex type and spokes together.

The data is stored as

    temp_kspace = temp_kspace.reshape((n_loc, nx, ny, 2 * ncoil))

And I think I would like to create another ataset name
This way I can always look back to the preivous dataset and maybe re-create some thing


So we will loop over the original dataset  (mixed) and loop over train/test/validation content. We ignore the train_25 etc
Load each file
Project the spokes with nufft
Test for reconstruction
Store the data

"""


class PrepData:
    def __init__(self, ddata):
        self.ddata = ddata
        self.file_list = os.listdir(ddata)
        self.file_name_list = [hmisc.get_base_name(x) for x in self.file_list]
        self.loaded_array = None
        self.trajectory = None
        self.file_index = None
        self.dplot = '/home/sharreve'
#
    def load_array(self, file_index):
        sel_file = self.file_list[file_index]
        sel_file_name = self.file_name_list[file_index]
        file_path = os.path.join(self.ddata, sel_file)
        loaded_array = hmisc.load_array(file_path, data_key='kspace')
        loaded_cpx_array = loaded_array[..., ::2] + 1j * loaded_array[..., 1::2]
        loaded_cpx_array = np.moveaxis(loaded_cpx_array, -1, 0)
        loaded_cpx_array = harray.scale_minmax(loaded_cpx_array, is_complex=True)
        # Returning (ncoil, nloc, ny, nx)
        self.loaded_array = loaded_cpx_array
        self.file_index = file_index
        self.sin_path = hmisc.find_file_in_dir(sel_file_name[:12], dir_name=DDATA_sin, ext='sin')
        return loaded_cpx_array
#
    def prep_loaded_array(self):
        # Perform fft on the last 2 dimensions
        self.loaded_array = np.fft.fft2(self.loaded_array)
        # Spoke definition
        N = self.loaded_array.shape[-1]
        self.trajectory = self.get_trajectory(N)
        self.loaded_array = np.fft.ifft2(np.fft.fftshift(np.fft.fft2(self.loaded_array), axes=(-2, -1)))
#
    @staticmethod
    def get_trajectory(N):
        max_spokes = int(np.ceil((np.pi * 2) * N))
        n_points = N
        img_size = (N, N)
        trajectory_radial = sigpy.mri.radial(coord_shape=(max_spokes, n_points, 2), img_shape=img_size)
        trajectory_radial = trajectory_radial.astype(np.float32)
        return trajectory_radial
#
    @staticmethod
    def sum_of_abs(x_array, sum_dim=0):
        return np.abs(x_array).sum(axis=sum_dim)
#
    def visualize_loaded_array(self):
        ddest_vis = os.path.join(self.dplot, self.file_name_list[self.file_index] + '.png')
        fig_obj = hplotc.ListPlot(self.sum_of_abs(self.loaded_array))
        fig_obj.figure.savefig(ddest_vis)
#
    def project_to_spoke(self, spoke_index, coil_index, card_index):
        sel_array = self.loaded_array[coil_index, card_index]
        sel_spoke = self.trajectory[spoke_index]
        undersampled_spoke = hobjrecon.undersample_img(sel_array, traj=sel_spoke)[0]
        undersampled_spoke_kspace = np.fft.ifftshift(np.fft.fft2(undersampled_spoke), axes=(-2, -1))
        return undersampled_spoke_kspace
#
    def project_to_spokes_mp(self):
        import multiprocessing as mp
        N = mp.cpu_count()
        ncoil, ncard = self.loaded_array.shape[:2]
        nspoke = self.trajectory.shape[0]
        cross_prod = list(itertools.product(range(nspoke), range(ncoil), range(ncard)))
        with mp.Pool(processes=N//4) as p:
            list_random_shim_results = p.map(self.project_to_spoke, cross_prod)
        return list_random_shim_results

data_type_list = ['train', 'test', 'validation']
ddata_mixed = os.path.join(DDATA, 'mixed')

i_datatype = data_type_list[0]
ddata_type = os.path.join(ddata_mixed, i_datatype, 'input')

prep_obj = PrepData(ddata_type)

for ii in range(len(prep_obj.file_list)):
    _ = prep_obj.load_array(ii)
    prep_obj.prep_loaded_array()
    prep_obj.visualize_loaded_array()
    hplotc.close_all()

prep_obj = PrepData(ddata_type)
loaded_array = prep_obj.load_array(0)
prep_obj.prep_loaded_array()
projected_spoke = prep_obj.project_to_spoke(0, 0, 0)
dsave = '/home/sharreve/local_scratch'

fig_obj = hplotc.ListPlot([projected_spoke, np.fft.fft2(projected_spoke)])
fig_obj.figure.savefig(os.path.join(dsave, 'singlespoke.png'))
