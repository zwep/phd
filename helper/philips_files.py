import pydicom


def get_TR_TE_dicom(x):
    pydicom_obj = pydicom.read_file(x, stop_before_pixels=True)
    TE = float(pydicom_obj[("2001", "1025")].value)
    TR_value = pydicom_obj[("2005", "1030")].value
    if isinstance(TR_value, list):
        TR = float(TR_value[0])
    elif isinstance(TR_value, float):
        TR = float(TR_value)
    else:
        TR = TR_value
    return TR, TE


def get_size_dict(x):
    pydicom_obj = pydicom.read_file(x, stop_before_pixels=True)
    slice_thickness = pydicom_obj.get(('0018', '0050'))
    pixel_spacing = pydicom_obj.get(('0028', '0030'))
    n_slices = pydicom_obj.get(('2001', '1018'))
    if slice_thickness is not None:
        slice_thickness = float(slice_thickness.value)
    if slice_thickness is not None:
        pixel_spacing = tuple([float(x) for x in pixel_spacing.value])
    if n_slices is not None:
        n_slices = n_slices.value

    FOV = None
    if pixel_spacing is not None:
        if len(pixel_spacing) == 2:
            nrows = pydicom_obj.get(('0028', '0010')).value
            ncols = pydicom_obj.get(('0028', '0011')).value
            FOV = (pixel_spacing[0] * nrows, pixel_spacing[1] * ncols)

    content_dict = {'slice_thickness': slice_thickness, 'pixel_spacing': pixel_spacing,
                    'fov': FOV, 'num_slices': n_slices}
    return content_dict
