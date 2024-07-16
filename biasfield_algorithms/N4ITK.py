import os
import helper.plot_fun as hplotf
import numpy as np
import helper.array_transf as harray
# N4ITK standard method...
import SimpleITK as sitk


def get_n4itk(x, mask=None, n_fit=4, n_iterations=100, output_biasfield=False):
    x = harray.scale_minmax(x)

    inputImage = sitk.GetImageFromArray(x)
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)

    if mask is None:
        # Mask creator is annoying...
        mask_array = harray.get_treshold_label_mask(x)
        maskImage = sitk.GetImageFromArray(mask_array.astype(int))
        maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)
    else:
        maskImage = sitk.GetImageFromArray(mask.astype(int))
        maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([n_iterations] * n_fit)
    res = corrector.Execute(inputImage, maskImage)
    output_n4itk_img = corrector.GetLogBiasFieldAsImage(inputImage)
    log_output_n4itk = sitk.GetArrayFromImage(output_n4itk_img)
    output_n4itk = np.exp(log_output_n4itk)

    output = x / output_n4itk

    if output_biasfield:
        return output, output_n4itk
    else:
        return output


if __name__ == "__main__":
    # Here we can test out this procedure....
    # Using measured data (read cpx)

    measured_path = '/home/bugger/Documents/data/7T/test_for_inhomog/prostate_7T'
    file_list = [os.path.join(measured_path, x) for x in os.listdir(measured_path)]
    sel_file_list = [x for x in file_list if x.endswith('npy')]

    for i_file in sel_file_list:
        A = np.load(i_file)
        A_abs = np.abs(A.sum(axis=0))
        A_res, A_mask = get_n4itk(A_abs, debug=True)
        hplotf.plot_3d_list([[A_abs, A_res, A_mask]])