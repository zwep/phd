from objective.inhomog_removal.PostProcClassic import PostProcClassic


# Homog prediction 8 channel
volunteer_dict = {'dest_dir': '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/t2w_n4itk',
                'img_dir': '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/t2w',
                'mask_dir': '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/body_mask'}

patient_dict = {'dest_dir': '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/image_n4itk',
                'img_dir': '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/image',
                'mask_dir': '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/mask'}

postproc_obj = PostProcClassic(storage_extension='nii', **patient_dict)
postproc_obj.run()
