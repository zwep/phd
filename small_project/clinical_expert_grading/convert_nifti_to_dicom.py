import os
import helper.plot_class as hplotc
import pydicom
import nibabel
import numpy as np
import helper.array_transf as harray
import helper.misc as hmisc

from data_prep.dataset.prostate.daan_reesink.order_of_slices import slice_order_dict
# Change the names to something more readable
selected_patient_files = {'7TMRI002': "7TMRI002",
                            '7TMRI003': "7TMRI003",
                            '7TMRI010': "7TMRI010",
                            '7TMRI015': "7TMRI015"}

selected_volunteer_files = {'pr_06012021_1647041_12_3_t2wV4': "7TMRI100",
                            'v9_03032021_1641286_9_3_t2wV4': "7TMRI101",
                            'v9_08072020_1827193_9_3_t2wV4': "7TMRI102",
                            'v9_10022021_1720565_11_3_t2wV4': "7TMRI103",
                            'v9_11022021_1635487_6_3_t2wV4': "7TMRI104",
                            'v9_18012021_0927536_6_3_t2wV4': "7TMRI105",
                            'v9_27052020_1611487_22_3_t2wV4': "7TMRI106"}

dicom_file_input = '/home/bugger/Documents/model_run/selection_model_results/7TMRI002nii/image.0001.dcm'
model_variation_list = ['single_biasfield', 'single_homogeneous']
for data_set in ['volunteer_corrected', 'patient_corrected']:
    for selected_model in model_variation_list:
        for data_type in ['pred', 'input']:
            dnifti = f'/home/bugger/Documents/model_run/selection_model_results/{selected_model}/{data_set}/{data_type}'
            ddest_dicom = f'/home/bugger/Documents/model_run/selection_model_results/{selected_model}/{data_set}/{data_type}_dicom'
            ddest_png = f'/home/bugger/Documents/model_run/selection_model_results/{selected_model}/{data_set}/{data_type}_png'
            if not os.path.isdir(ddest_dicom):
                os.makedirs(ddest_dicom)
            if not os.path.isdir(ddest_png):
                os.makedirs(ddest_png)
            dicom_obj = pydicom.read_file(dicom_file_input)
            max_value = dicom_obj.pixel_array.max()
            # Get the DICOM object..
            for nifti_file in sorted(os.listdir(dnifti)):
                nifti_file_path = os.path.join(dnifti, nifti_file)
                base_name = hmisc.get_base_name(nifti_file)
                dicom_file_path = os.path.join(ddest_dicom, base_name + '.dcm')
                if data_set == 'volunteer_corrected':
                    if base_name not in selected_volunteer_files:
                        continue
                    else:
                        dicom_file_path = os.path.join(ddest_dicom, selected_volunteer_files[base_name] + '.dcm')
                        dicom_png_file_path = os.path.join(ddest_png, selected_volunteer_files[base_name] + '.png')
                if data_set == 'patient_corrected':
                    if base_name not in selected_patient_files:
                        continue
                    else:
                        dicom_file_path = os.path.join(ddest_dicom, selected_patient_files[base_name] + '.dcm')
                        dicom_png_file_path = os.path.join(ddest_png, selected_patient_files[base_name] + '.png')
                loaded_nifti = hmisc.load_array(nifti_file_path)
                loaded_nifti = loaded_nifti.T[:, ::-1, ::-1]
                if base_name in slice_order_dict:
                    slice_order_index = slice_order_dict[base_name]
                    loaded_nifti = loaded_nifti[slice_order_index]
                dicom_obj.Rows, dicom_obj.Columns = loaded_nifti.shape[-2:]
                dicom_obj.NumberOfFrames = loaded_nifti.shape[0]
                corrected_result = harray.scale_minmax(loaded_nifti, axis=(-2, -1))
                corrected_result = (max_value * corrected_result).astype(np.int16)
                print('Shape of data ', corrected_result.shape)
                print(f' N slices {dicom_obj.NumberOfFrames} N rows {dicom_obj.Rows} N cols {dicom_obj.Columns}')
                dicom_obj.PixelData = corrected_result.tobytes()
                # First remove the file...
                if os.path.isfile(dicom_file_path) or os.path.islink(dicom_file_path):
                    os.remove(dicom_file_path)
                dicom_obj.save_as(dicom_file_path)
                n_chan = corrected_result.shape[0]
                fig_obj = hplotc.ListPlot(corrected_result[n_chan//2], ax_off=True)
                fig_obj.figure.savefig(dicom_png_file_path, bbox_inches='tight', pad_inches=0.0)
                hplotc.close_all()
        #
        # dicom_obj = pydicom.read_file(dicom_file_path)
        # dir(dicom_obj)
        #
        # dicom_obj.SeriesInstanceUID
        #
