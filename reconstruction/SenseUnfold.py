
import multiprocessing as mp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import skimage.transform as sktransform
import scipy.signal
import scipy.ndimage
import numpy as np
from typing import List, Union, Tuple
import helper.array_transf as harray
import multiprocessing


# Used only inside this function..
def _calc_inv(container):
    temp_sens, temp_signal = container
    temp_rho = np.matmul(np.linalg.pinv(temp_sens), temp_signal)
    return temp_rho


class SenseUnfold:
    def __init__(self, reference_obj, n_coils=8, pre_loaded_image=None):
        self.n_interp = None
        self.n_coils = n_coils
        self.refscan_coil_names = reference_obj.sub_index_col
        if pre_loaded_image is None:
            refscan = reference_obj.get_cpx_img()
            self.refscan, self.dimension_reference = self._prep_reference_scan(refscan)
        else:
            pre_loaded_image = np.moveaxis(pre_loaded_image, -2, -1)
            self.refscan = pre_loaded_image
            self.dimension_reference = pre_loaded_image.shape

        reference_parameters = reference_obj.get_par_file()
        self.off_centre_ref, self.angulation_ref, self.fov_ref = self.extract_parameters(reference_parameters)
        # voxel_ref = [mm] / [voxel]
        self.voxel_ref = (self.fov_ref / self.dimension_reference)

    def _prep_reference_scan(self, refscan):
        # Puts every dimension on the right axis
        ref_z = refscan.shape[self.refscan_coil_names.index('slice')]
        ref_x, ref_y = refscan.shape[-2:]
        # This should be in the order (ap, fh, lr)
        # This already shows that we need to switch the last two axes..
        dim_ref = [ref_z, ref_y, ref_x]
        # Select the first "location" of the refscan
        refscan_sel = np.squeeze(refscan[:, 0])
        # Change the order of the last two axes
        # This brings it to (coils, ap, fh, lr)
        refscan_sel = np.moveaxis(refscan_sel, -2, -1)
        # Select only the n-coils...
        refscan_sel = refscan_sel[-self.n_coils:]
        return refscan_sel, dim_ref

    @staticmethod
    def unfold_mp(folded_image, reference_img, folding_factor, axis=-2):

        # Make complex here optional based on the input..
        unfolded_image = np.zeros(reference_img.shape[-2:], dtype=complex)

        if axis == -1:
            reference_img = np.swapaxes(reference_img, -2, -1)
            folded_image = np.swapaxes(folded_image, -2, -1)

        n_channel, n_x, n_y = reference_img.shape
        n_x_fold = int(n_x / folding_factor)

        res = []
        for i_x in range(n_x_fold):
            for i_y in range(n_y):
                temp_signal = folded_image[:, i_x, i_y]
                temp_sens = reference_img[:, i_x::n_x_fold, i_y]
                res.append((temp_sens, temp_signal))

        # I dont know any more adanvced way to calculate all these matrix inversion...
        N = multiprocessing.cpu_count()
        with mp.Pool(processes=N) as p:
            results = p.map(_calc_inv, res)

        counter = 0
        for i_x in range(n_x_fold):
            for i_y in range(n_y):
                unfolded_image[i_x::n_x_fold, i_y] = results[counter]
                counter += 1

        if axis == -1:
            unfolded_image = np.swapaxes(unfolded_image, -2, -1)

        return unfolded_image

    # Unfolding...
    @staticmethod
    def unfold(folded_image, reference_img, folding_factor, axis=-2):
        # Make complex here optional based on the input..
        unfolded_image = np.zeros(reference_img.shape[-2:], dtype=complex)

        if axis == -1:
            reference_img = np.swapaxes(reference_img, -2, -1)
            folded_image = np.swapaxes(folded_image, -2, -1)

        n_channel, n_x, n_y = reference_img.shape
        n_x_fold = int(n_x / folding_factor)

        for i_x in range(n_x_fold):
            for i_y in range(n_y):
                temp_signal = folded_image[:, i_x, i_y]
                temp_sens = reference_img[:, i_x::n_x_fold, i_y]
                temp_rho = np.matmul(np.linalg.pinv(temp_sens), temp_signal)
                unfolded_image[i_x::n_x_fold, i_y] = temp_rho

        if axis == -1:
            unfolded_image = np.swapaxes(unfolded_image, -2, -1)

        return unfolded_image

    @staticmethod
    def extract_parameters(parameter_file):
        # Extract offset in one direction....
        # Every array here is ordered in (ap, fh, lr) direction
        # This `next` gets the first index of the array with indexing it like [0]
        off_centre = next((np.array(v.split()).astype(float) for k, v in parameter_file.items() if 'Off Centre' in k), None)
        angulation = next((np.array(v.split()).astype(float) for k, v in parameter_file.items() if 'Angulation' in k), None)
        fov = next((np.array(v.split()).astype(float) for k, v in parameter_file.items() if 'FOV' in k), None)

        return off_centre, angulation, fov

    @staticmethod
    def _visualize_plane(XYZ, acq_plane_rot, acq_plane_rot_trans):
        # Function that is only used WITHIN another to visualize intermediate steps
        # Visualize plane..
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y, Z = XYZ
        ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c='b', label='Non rotated plane')
        ax.scatter(acq_plane_rot[:, 0], acq_plane_rot[:, 1], acq_plane_rot[:, 2], c='r', label='Rotated plane')
        ax.scatter(acq_plane_rot_trans[:, 0], acq_plane_rot_trans[:, 1], acq_plane_rot_trans[:, 2], c='k',
                   label='Rotated and translated plane')
        ax.set_xlabel('ap')
        ax.set_ylabel('fh')
        ax.set_zlabel('lr')
        plt.legend()

    def get_coordinate_acq_plane(self, acq_parameters, debug=False):
        """
        This returns a vector that shows where the location of the scan was with respect to the REF SCAN.
        The vector (`offset`) is range of min..max value of pixel values in the REF SCAN.
        The vector (`midpoint`) is the translation
        """
        reference_ap, reference_fh, reference_lr = self.dimension_reference

        off_centre_cine, angulation_cine, fov_cine = self.extract_parameters(acq_parameters)
        # We want to set each slice thickness to 8mm
        # The amount of locations will be fixed in the next function
        # Feels a bit sketchy though, but might be useful
        index_min_fov = np.argmin(fov_cine)
        fov_cine[index_min_fov] = 8.0

        # Distance compared to mid slice ref scan. [voxel] = [mm] / ([mm] / [voxel])
        comp_ap, comp_fh, comp_lr = ((self.off_centre_ref - off_centre_cine) / self.voxel_ref)
        # Middle of the FOV [in voxels]
        mid_point = (reference_ap / 2 - comp_ap,  reference_fh / 2 - comp_fh, reference_lr / 2 - comp_lr)
        # Distance from the middle to the edges...
        # [voxel] = [mm] / ([mm] / [voxel])
        fov_distance_array = (fov_cine / self.voxel_ref) / 2
        boundary_coord = np.array([-fov_distance_array, fov_distance_array])

        if debug:
            print('FOV of acq            ', fov_cine)
            print('Voxel size refscan    ', self.voxel_ref)
            print('Distance from center  ', fov_distance_array)
            print('Boundary coord       \n', boundary_coord, end='\n\n')
            print('Off-center difference ', self.off_centre_ref - off_centre_cine)
            print('Compensation of mid   ', comp_ap, comp_fh, comp_lr)
            print('Mid point             ', mid_point)

        return boundary_coord, mid_point

    def get_acq_plane(self, acq_parameter,
                      center_coord,
                      boundary_coord,
                      n_interp: Union[Tuple[int, ...], int] = 100,
                      R_total=None):
        """
        :param center_coord:
        :param boundary_coord:
        :param angulation_acq:
        :param n_interp: should be an interger or list of integers (dim 3)
        :param visualize:
        :return:
        """
        # Use n_intepr if it is chosen
        if self.n_interp is not None:
            n_interp = n_interp

        if isinstance(n_interp, int):
            n_interp = [n_interp] * 3

        # We need angulation to rotate the whole slice
        # We need the original FOV to see how large the slice-direction is.
        # We need to off center to
        off_centre_acq, angulation_acq, fov_acq = self.extract_parameters(acq_parameter)

        # Now create one line for each direction. Approximate this line by 100 points.
        # This number of points can be increased
        single_line_coord = []
        for i, xy in enumerate(zip(*boundary_coord)):
            # Check if this helps with the final orientation....
            x, y = xy
            x, y = sorted([x, y])
            if x - y != 0:
                temp = np.linspace(x, y, n_interp[i])
            else:
                temp = np.linspace(x, y, 1)

            single_line_coord.append(temp)

        # With these single lines, we can then create a mesh grid
        # It might be that the indexing goes the way we want it now..
        X, Y, Z = np.meshgrid(*single_line_coord, indexing='ij')
        final_shape = X.shape

        R_ap = harray.rot_x(angulation_acq[0])
        R_fh = harray.rot_y(angulation_acq[1])
        R_rl = harray.rot_z(angulation_acq[2])

        # After a lot of trial and error.. this should be the orientation that is used..
        if R_total is None:
            R_total = R_rl @ R_ap @ R_fh

        print('Rotation matrix \n', R_total)
        # Rotate the whole meshgrid we just created
        acq_plane_rot = []
        for ix, iy, iz in zip(X.ravel(), Y.ravel(), Z.ravel()):
            temp = [ix, iy, iz]
            rot_temp = R_total @ temp
            acq_plane_rot.append(rot_temp)

        acq_plane_rot = np.array(acq_plane_rot)

        # Now.. create slices for each location
        n_loc = int(acq_parameter['Max. number of slices/locations'])
        index_min_fov = np.argmin(fov_acq)
        center_line = np.zeros((3, n_loc))
        center_line[index_min_fov, :] = np.linspace(-fov_acq[index_min_fov] // 2 + 4, fov_acq[index_min_fov] // 2 - 4, n_loc)
        center_rot_ref = (R_total @ center_line / self.voxel_ref[:, None]).T

        acq_plante_rot_trans_loc = []
        print('-- Following lines can be checked in the .sin files')
        print('Offset due to location in voxels')
        for i_loc in range(n_loc):
            print('\t\t', center_rot_ref[i_loc] * self.voxel_ref + off_centre_acq)
            # Shift the rotated plane with the mid point we calculated before
            acq_plane_rot_trans = acq_plane_rot + np.array(center_coord).reshape(1, 3) + center_rot_ref[i_loc]
            acq_plante_rot_trans_loc.append(acq_plane_rot_trans)

        acq_plante_rot_trans_loc = np.stack(acq_plante_rot_trans_loc, axis=0)

        return acq_plante_rot_trans_loc, final_shape

    def map_acq_plane_coords(self, acquisition_slice, final_shape):
        coil_img_sliced = []
        print('Debug... shape during map acq plane ', self.refscan.shape, acquisition_slice.shape)
        for x_real, x_imag in zip(self.refscan.real, self.refscan.imag):
            temp_real = scipy.ndimage.map_coordinates(x_real, acquisition_slice.T).reshape(final_shape)
            temp_imag = scipy.ndimage.map_coordinates(x_imag, acquisition_slice.T).reshape(final_shape)
            temp_cpx = temp_real + 1j * temp_imag
            coil_img_sliced.append(temp_cpx)

        coil_img_sliced = np.array(coil_img_sliced)
        return coil_img_sliced

    @staticmethod
    def resize_acq_plane(refscan_slice, dim_cine_sense):
        # Input dimension is (coil, nx, ny)
        # Dim cine sense is (nx, ny)
        resized_refscan = []
        for x_real, x_imag in zip(refscan_slice.real, refscan_slice.imag):
            temp_real = sktransform.resize(x_real, dim_cine_sense, preserve_range=True)
            temp_imag = sktransform.resize(x_imag, dim_cine_sense, preserve_range=True)
            temp_cpx = temp_real + 1j * temp_imag
            resized_refscan.append(temp_cpx)

        resized_refscan = np.array(resized_refscan)
        return resized_refscan

    def get_refscan_acq(self, acq_param, target_size):
        # Need to figure out these things...
        prep_direction = acq_param['Preparation direction']

        boundary_coord, center_coord = self.get_coordinate_acq_plane(acq_param, debug=False)
        _, _, fov_acq = self.extract_parameters(acq_param)
        index_min_fov = np.argmin(fov_acq)

        # I assume that this should be applied in the SENSE direction
        # This might coincide with the preperation direction
        if prep_direction == 'RL':
            boundary_coord[:, 2] = boundary_coord[:, 2] * 1.5
        if prep_direction == 'FH':
            boundary_coord[:, 1] = boundary_coord[:, 1] * 1.5
        if prep_direction == 'AP':
            boundary_coord[:, 0] = boundary_coord[:, 0] * 1.5

        acquisition_plane, final_shape = self.get_acq_plane(acq_parameter=acq_param,
                                                            boundary_coord=boundary_coord,
                                                            center_coord=center_coord,
                                                            n_interp=100)

        ref_acq_plane = []
        for acquisition_plane_loc in acquisition_plane:
            acq_plane = self.map_acq_plane_coords(acquisition_slice=acquisition_plane_loc, final_shape=final_shape)
            acq_plane_mean = np.mean(acq_plane, axis=index_min_fov + 1)

            if prep_direction == 'RL':
                # Is this... what we want... or what we need...?
                print('Rotating refscan due to RL prep direction')
                acq_plane_mean = np.swapaxes(acq_plane_mean, 1, 2)  # Needed for 2ch thing

            acq_plane_resized = self.resize_acq_plane(acq_plane_mean, target_size)
            acq_plane_rescaled = acq_plane_resized / np.abs(acq_plane_resized).sum(axis=0)

            ref_acq_plane.append(acq_plane_rescaled)

        ref_acq_plane = np.array(ref_acq_plane)
        return ref_acq_plane


if __name__ == "__main__":
    import ReadCpx as read_cpx

    sense_file = '/media/bugger/MyBook/data/data_for_gyrotools/v9_02052021_1228480_4_3_senserefscanclassicV4.cpx'
    folded_file = '/media/bugger/MyBook/data/data_for_gyrotools/v9_02052021_1244361_16_3_4chV4.cpx'

    cpx_sense_obj = read_cpx.ReadCpx(sense_file)
    cpx_folded_obj = read_cpx.ReadCpx(folded_file)
    folded_img = cpx_folded_obj.get_cpx_img()
    folded_param = cpx_folded_obj.get_par_file()

    # I know these attributes because I scanned it myself..
    sel_number_coils = 8
    sense_factor = 3
    cine_x, cine_y = folded_img.shape[-2:]
    target_size = (cine_x * sense_factor, cine_y)

    # Select one cardiac phase from the folded image
    sel_folded_img = np.squeeze(folded_img)[-sel_number_coils:, 0]

    # Create the SenseUnfold object and get the refscan slice
    sense_obj = SenseUnfold(cpx_sense_obj, n_coils=sel_number_coils)
    ref_acq_plane = sense_obj.get_refscan_acq(acq_param=folded_param, target_size=target_size)
    index_nan = np.isnan(ref_acq_plane)
    if index_nan.sum():
        ref_acq_plane[index_nan] = 0

    # Visualize the sliced refscan with help of the parameters in `folded_param`
    fig, ax = plt.subplots(sel_number_coils)
    ax = ax.ravel()
    for i, i_img in enumerate(np.squeeze(ref_acq_plane)):
        ax[i].imshow(np.abs(i_img[:, ::-1]))

    # Visualize both folded and refscan images together
    # From this we see that we need to flip the acquired refscan...
    fig, ax = plt.subplots(3)
    ax[0].imshow(np.abs(sel_folded_img).sum(axis=0))
    ax[1].imshow(np.abs(ref_acq_plane[0].sum(axis=0)))
    ax[2].imshow(np.abs(ref_acq_plane[0, :, :, ::-1].sum(axis=0)))

    # The function below is able to unfold the folded image. BUT we might need to allign
    # the refscan slice correctly. This requires manual labor, but can be worth it.
    unfolded_img = sense_obj.unfold_mp(sel_folded_img, ref_acq_plane[0, :, :, ::-1], sense_factor)

    # Here we can see the unfolded scan.
    # Masking can improve the image
    fig, ax = plt.subplots(1)
    ax.imshow(np.abs(unfolded_img))