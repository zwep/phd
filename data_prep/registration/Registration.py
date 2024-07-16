"""
Created a registration class to make this process more smooth and better for reproducibility


"""
import nrrd
import os
import numpy as np
import scipy.io
import h5py
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import SimpleITK as sitk
import helper.misc as hmisc


class Registration:
    """
    Creating a class that can register image A on B using as example A_mask and B_mask
    """
    def __init__(self, A, B, A_mask=None, B_mask=None, **kwargs):
        # Assume size (ny, nx) for all images...?
        self.debug = kwargs.get('debug', False)
        self.scan_type = kwargs.get('scan_type', None)
        self.registration_options = kwargs.get('registration_options', '')
        self.A = A
        self.B = B
        self.A_mask = A_mask
        self.B_mask = B_mask
        if self.A_mask is None:
            self.A_mask = harray.get_treshold_label_mask(A)
            self.A_mask = self.A_mask.astype(int)

        if self.B_mask is None:
            self.B_mask = harray.get_treshold_label_mask(B)
            self.B_mask = self.B_mask.astype(int)

        self.registration_parameter = self.get_registration_parameter()
        self.registration_mapping = None

    def get_registration_parameter(self):
        # Build a list with registration parameters
        registration_parameter = sitk.VectorOfParameterMap()
        rigid_map = sitk.GetDefaultParameterMap("rigid")
        affine_map = sitk.GetDefaultParameterMap("affine")
        bspline_map = sitk.GetDefaultParameterMap("bspline")

        if 'rigid' in self.registration_options:
            if self.debug:
                print('Adding rigid registration parameters')
            registration_parameter.append(rigid_map)
        if 'affine' in self.registration_options:
            if self.debug:
                print('Adding Affine registration parameters')
            registration_parameter.append(affine_map)
        if 'bspline' in self.registration_options:
            if self.debug:
                print('Adding Affine bspline parameters')
            registration_parameter.append(bspline_map)

        return registration_parameter

    # This is about the masks....
    def register_mask(self):
        """
        We map A_mask onto B_mask
        """
        elastix_obj = sitk.ElastixImageFilter()
        if not self.debug:
            elastix_obj.LogToConsoleOff()

        # Convert to Image object
        moving_image = sitk.GetImageFromArray(self.A_mask)
        fixed_image = sitk.GetImageFromArray(self.B_mask)

        elastix_obj.SetMovingImage(moving_image)
        elastix_obj.SetFixedImage(fixed_image)

        elastix_obj.SetParameterMap(self.registration_parameter)

        elastix_obj.Execute()

        transform_mapping = elastix_obj.GetTransformParameterMap()
        self.registration_mapping = transform_mapping

        return transform_mapping

    def apply_registration(self, x):
        # Check if the transformation from mask A to mask B goes like we want..
        # We return
        if self.registration_mapping is not None:
            x_image = sitk.GetImageFromArray(x)
            result_image = sitk.Transformix(x_image, self.registration_mapping)
            result_array = sitk.GetArrayFromImage(result_image)

            return result_array
        else:
            print('Run registration first: .register_mask()')

    @staticmethod
    def correct_image(x, mask):
        # Correct with mask
        # Make sure that we have no troubles with really tiny numbers
        x_array = x * mask
        mask_int = ((1 - mask) == 1).astype(int)
        input_array_close = np.isclose(x_array, 0).astype(int)
        input_array_outside = (input_array_close * mask_int).astype(bool)
        x_array[input_array_outside] = 0
        return x_array

    def apply_registration_cpx(self, x):
        # Check if the transformation from mask A to mask B goes like we want..
        # We return
        if self.registration_mapping is not None:
            x_reg_real = self.apply_registration(x.real)
            x_reg_imag = self.apply_registration(x.imag)
            return x_reg_real + 1j * x_reg_imag
        else:
            print('Run registration first: .register_mask()')

    def validate_mask_mapping(self):
        # Check if the transformation from mask A to mask B goes like we want..
        # We return
        if self.registration_mapping is not None:
            A_mask_image = sitk.GetImageFromArray(self.A_mask)
            B_mask_image = sitk.Transformix(A_mask_image, self.registration_mapping)
            B_mask_array = sitk.GetArrayFromImage(B_mask_image)

            dice_score = hmisc.dice_metric(self.B_mask, B_mask_array)
            return B_mask_array, dice_score
        else:
            print('Run registration first: .register_mask()')

    def display_mask_validation(self):
        if self.registration_mapping is not None:
            B_mask_approx, dice_score = self.validate_mask_mapping()
            fig_handle = hplotf.plot_3d_list([[self.A, self.A_mask, B_mask_approx, self.B_mask]],
                                             subtitle=[['starting image', 'starting mask',
                                                        'approximation mask', 'target mask']])
            temp_title = f'dice score {dice_score:.3f}'
            fig_handle.suptitle(temp_title)
            return fig_handle
        else:
            print('Run registration first: .register_mask()')

    def display_registration(self, x):
        if self.registration_mapping is not None:
            result_image = self.apply_registration(x)
            fig_handle = hplotf.plot_3d_list([[x, result_image]],
                                             subtitle=[['starting image', 'result transformation']])
            return fig_handle
        else:
            print('Run registration first: .register_mask()')

    def display_content(self):
        fig_handle = hplotf.plot_3d_list([[self.A, self.A_mask, self.B, self.B_mask]],
                                         subtitle=[['starting image', 'starting mask', 'target image', 'target mask']])
        return fig_handle


if __name__ == "__main__":
    import scipy.ndimage
    import skimage.transform
    import skimage.data

    A_image = skimage.data.astronaut().sum(axis=-1)

    n_x = 33
    n_delta = 5
    A = np.zeros((n_x, n_x))
    A[n_x//2 - n_delta:n_x//2 + n_delta, n_x//2 - n_delta:n_x//2 + n_delta] = 1
    A = scipy.ndimage.rotate(A, 23)
    A = skimage.transform.rescale(A, 3)
    B = np.zeros(A.shape)
    n_x = B.shape[0]
    B[n_x//2 - n_delta:n_x//2 + n_delta, n_x//2 - n_delta:n_x//2 + n_delta] = 1

    A_image = skimage.transform.resize(A_image, B.shape)

    hplotf.plot_3d_list([A, B, A_image])

    register_obj = Registration(B, A, A_mask=B, B_mask=A, register_obj='rigidaffine')
    register_obj.debug = True
    register_obj.register_mask()
    register_obj.validate_mask_mapping()
    register_obj.display_mask_validation()

    res_image = register_obj.apply_registration(A_image * B)
    hplotf.plot_3d_list([res_image, A_image])

