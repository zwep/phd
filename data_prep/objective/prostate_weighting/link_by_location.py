
"""
The idea is to check whether we can align the MRL/MRI data by location given in the header
Is this one reliable..?

It does not feel reliable...
Way too annoying to check everything.

I re-used some code from the `dicom_to_h5_file_order` just to take things apart.
"""

import itertools
import pydicom
import os
import helper.plot_class as hplotc
ddata_MRL = '/home/bugger/Documents/data/compare_astrid_data/dicom_1p5T'
ddata_MRI = '/home/bugger/Documents/data/compare_astrid_data/dicom_3T'


def get_dates_with_files(ddir):
    # This returns the file names, dicom object and associated dates
    MR_file_list = [os.path.join(ddir, x) for x in os.listdir(ddir) if x.endswith('.dcm')]
    dicom_obj_list = [pydicom.read_file(x, stop_before_pixels=True) for x in MR_file_list]
    # Get all the dates from the header
    extracted_dates = [(MR_file_list[i], x, x[('0008', '0012')].value) for i, x in enumerate(dicom_obj_list)]
    unique_dates = list(set([x[2] for x in extracted_dates]))
    return extracted_dates, unique_dates


def get_sorted_file_list(file_info, sel_date):
    # Sirt sorts the file obtained fmor the previous method for a specific date
    # This sort is now based on the acquisition number and instance number
    sel_date_file_list, sel_date_dicom_list = zip(*[x[:2] for x in file_info if x[2] == sel_date])

    # Get all the acquisition numbers and instance numbers from the files in the directory..
    res_acq = [x.get(('0020', '0012'), None) for x in sel_date_dicom_list]
    num_acq = [x.value if x else None for x in res_acq]
    res_inst = [x.get(('0020', '0013'), None) for x in sel_date_dicom_list]
    num_inst = [x.value if x else None for x in res_inst]
    file_list_dict = [{'acq_num': x, 'inst_num': y, 'file_name': k} for k, x, y in
                      zip(sel_date_file_list, num_acq, num_inst)]
    # We need to remove the Nones here
    file_list_dict = [x for x in file_list_dict if x['acq_num'] is not None]
    sorted_file_list_dict = sorted(file_list_dict, key=lambda x: x['acq_num'])
    sorted_file_dict = {}
    for acq_num, group in itertools.groupby(sorted_file_list_dict, key=lambda x: x['acq_num']):
        list_group = list(group)
        # print('\t\t', acq_num, len(list_group))
        inst_num_sorted = sorted(list_group, key=lambda x: x['inst_num'])
        sorted_file_list = [x['file_name'] for x in inst_num_sorted]
        sorted_file_dict.setdefault(acq_num.original_string, [])
        sorted_file_dict[acq_num.original_string].extend(sorted_file_list)
    return sorted_file_dict


def show_example_sorted_file(sorted_files):
    # Here we can show a simple example of
    plot_array = []
    title_array = []
    for k, v in sorted_files.items():
        n_files = len(v)
        sel_file = v[n_files//2]
        dicom_obj = pydicom.read_file(sel_file)
        plot_array.append(dicom_obj.pixel_array)
        title_array.append(k)

    fig_obj = hplotc.ListPlot(plot_array, subtitle=title_array)
    return fig_obj

# Get all the files and their dates AND the unique dates
# With this we can make a single selection of files we want
MRL_files_dates, MRL_dates = get_dates_with_files(ddata_MRL)
MRI_files_dates, MRI_dates = get_dates_with_files(ddata_MRI)

# SElect the date...
sel_date = MRI_dates[0]
file_info = MRI_files_dates

# Get the sorted version of these files
# This couldve been done better with a table and a groupby/sortby operation
MRI_sorted_files = get_sorted_file_list(MRI_files_dates, MRI_dates[0])
MRL_sorted_files = get_sorted_file_list(MRL_files_dates, MRL_dates[0])

# Show an example...
show_example_sorted_file(MRI_sorted_files)
show_example_sorted_file(MRL_sorted_files)

sel_MRI_item = '8'
sel_MRI_files = MRI_sorted_files[sel_MRI_item]
sel_MRI_array = [pydicom.read_file(x).pixel_array for x in sel_MRI_files]

sel_MRL_item = '3'
sel_MRL_files = MRL_sorted_files[sel_MRL_item]
sel_MRL_array = [pydicom.read_file(x).pixel_array for x in sel_MRL_files]

# VIsual inspection... The current decisions shows that we have really EQUAL files
# Now check whether the are also the same in terms of this position..
import numpy as np
hplotc.SlidingPlot(np.array(sel_MRI_array))
hplotc.SlidingPlot(np.array(sel_MRL_array))

# Now check the distance...
def get_position_array(sel_files):
    position_array = []
    for x in sel_files:
        dicom_obj = pydicom.read_file(x, stop_before_pixels=True)
        float_position = list(map(float, dicom_obj[('0020', '0032')].value))
        position_array.append(float_position)

    return position_array



for sel_date in MRL_dates:
    MRL_sorted_files = get_sorted_file_list(MRL_files_dates, sel_date)
    for sel_MRL_item, sel_MRL_files in MRL_sorted_files.items():
        dicom_obj = pydicom.read_file(sel_MRL_files[0], stop_before_pixels=True)
        int(dicom_obj['0018', '0088'].value)
        sel_MRL_array = [pydicom.read_file(x).pixel_array for x in sel_MRL_files]
        n_files = len(sel_MRL_array)

        MRL_position = get_position_array(sel_MRL_files)
        MRL_position = np.array(MRL_position)
        subtitle = [MRL_position[0], MRL_position[n_files // 2], MRL_position[-1]]

        subtitle = [['; '.join([str(round(x, 2)) for x in y])] for y in subtitle]
        hplotc.ListPlot([sel_MRL_array[0], sel_MRL_array[n_files // 2], sel_MRL_array[-1]],
                        subtitle=subtitle, ax_off=True)

        # MRL_position = MRL_position - MRL_position[0]

        xpos, ypos, zpos = zip(*MRL_position)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xpos, ypos, zpos, c='r')
        title_string = f'date {sel_date} item {sel_MRL_item}'
        fig.suptitle(title_string)


for sel_date in MRI_dates:
    MRI_sorted_files = get_sorted_file_list(MRI_files_dates, sel_date)
    for sel_MRI_item, sel_MRI_files in MRI_sorted_files.items():
        dicom_obj = pydicom.read_file(sel_MRI_files[0])
        int(dicom_obj['0018', '0088'].value)
        sel_MRI_array = [pydicom.read_file(x).pixel_array for x in sel_MRI_files]
        MRI_position = get_position_array(sel_MRI_files)
        MRI_position = np.array(MRI_position)
        # MRI_position = MRI_position - MRI_position[0]
        n_files = len(sel_MRI_array)
        title_string = f'date {sel_date} item {sel_MRI_item}'
        subtitle = [MRI_position[0], MRI_position[n_files // 2], MRI_position[-1]]
        subtitle = [['; '.join([str(round(x, 2)) for x in y])] for y in subtitle]
        hplotc.ListPlot([sel_MRI_array[0], sel_MRI_array[n_files // 2], sel_MRI_array[-1]],
                        subtitle=subtitle, ax_off=True, title=title_string)

        xpos, ypos, zpos = zip(*MRI_position)
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xpos, ypos, zpos, c='k')

        fig.suptitle(title_string)

"""
Now use the compare array thing to determine which is the best 
"""

from data_prep.objective.prostate_weighting.extract_MRI_images import CompareImages

patient_file_num_zero_list = []
for sel_date in MRI_dates:
    MRI_sorted_files = get_sorted_file_list(MRI_files_dates, sel_date)
    for sel_MRI_item, sel_MRI_files in MRI_sorted_files.items():
        n_slice = len(sel_MRI_files)
        dicom_obj = pydicom.read_file(sel_MRI_files[n_slice//2])
        int(dicom_obj['0018', '0088'].value)
        slice_orientation = dicom_obj['2001', '100b'].value
        if slice_orientation == 'TRANSVERSAL':
            mid_slice_zeros = (dicom_obj.pixel_array == 0).mean()
        else:
            print('No transversal slice.. but: ', slice_orientation)
            continue

        # We want atleast 10% to be totally black.. otherwise we have nothing useful
        if mid_slice_zeros > 0.1:
            patient_file_num_zero_list.append((sel_date, sel_MRI_item, n_slice, mid_slice_zeros))

temp_sel = sorted(patient_file_num_zero_list, key=lambda x: x[-1])[::-1]
sel_date, sel_item, max_slice_MRI, score = temp_sel[0]
selMRI_sorted_files = get_sorted_file_list(MRI_files_dates, sel_date)
final_MRI_files = selMRI_sorted_files[sel_item]
final_MRI_array = [pydicom.read_file(x).pixel_array for x in final_MRI_files]
# Print some statistics about the selected file...
dicom_obj = pydicom.read_file(final_MRI_files[0])
print(dicom_obj['0018', '0088'])
print(dicom_obj['0020', '0032'])
print('Max slices', len(final_MRI_files))

compare_score_list_mask = []
compare_score_list_ssim = []
for sel_date in MRL_dates:
    MRL_sorted_files = get_sorted_file_list(MRL_files_dates, sel_date)
    for sel_MRL_item, sel_MRL_files in MRL_sorted_files.items():
        dicom_obj = pydicom.read_file(sel_MRL_files[0], stop_before_pixels=True)
        print(dicom_obj['0018', '0088'])
        print('Max slices', len(final_MRI_files))

        sel_MRL_array = [pydicom.read_file(x).pixel_array for x in sel_MRL_files]

        # Compare to middle slice from MRI..?
        reference_slice = final_MRI_array[len(final_MRI_array)//2]
        sel_MRL_array_compare = [np.sqrt(((x - reference_slice)**2).sum(axis=0)) for x in sel_MRL_array]


        compare_obj = CompareImages(final_MRI_files, sel_MRL_array)

        # This result the average SSIM over all the slices.
        # I think SSIM is slower thank comparing mask values
        compare_score_ssim = compare_obj.compare_array()
        compare_score_mask = compare_obj.compare_mask_array()
        compare_score_list_ssim.append((sel_date, sel_MRL_item, len(sel_MRL_files), compare_score_ssim))
        compare_score_list_mask.append((sel_date, sel_MRL_item, len(sel_MRL_files), compare_score_mask))


temp_sel = sorted(compare_score_list_mask, key=lambda x: (-x[-1], x[-2]))[::-1][6]
sel_date, sel_item, max_slice_MRL, score = temp_sel
selMRL_sorted_files = get_sorted_file_list(MRL_files_dates, sel_date)
final_MRL_files = selMRL_sorted_files[sel_item]
final_MRL_array = [pydicom.read_file(x).pixel_array for x in final_MRL_files]
dicom_obj = pydicom.read_file(final_MRL_files[54])
dicom_obj['0018', '0088']
float_position = list(map(float, dicom_obj[('0020', '0032')].value))
hplotc.SlidingPlot(np.array(final_MRI_array))
hplotc.SlidingPlot(np.array(final_MRL_array))