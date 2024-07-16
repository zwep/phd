import numpy as np
import os
import re
import csv
import helper.plot_class as hplotc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

ddest = '/local_scratch/sharreve/mri_data/response_google_forms'
dcsv = os.path.join(ddest, 'bertine_joost_Review prostaat beelden.csv')
dimg = os.path.join(ddest, 'wrong_images')

list_model_B_dict = []
list_model_H_dict = []
with open(dcsv, 'r') as f:
    csv_obj = csv.reader(f)
    header = next(csv_obj)
    del header[0]
    n_questions = len(header)
    model_B = header[:n_questions // 2]
    model_H = header[n_questions // 2:]
    for i_row in csv_obj:
        del i_row[0]
        model_B_dict = dict(zip(model_B, i_row[:n_questions // 2]))
        model_H_dict = dict(zip(model_H, i_row[n_questions // 2:]))
        list_model_B_dict.append(model_B_dict)
        list_model_H_dict.append(model_H_dict)

# Create the paths where the images will be located...
dmodel = '/local_scratch/sharreve/model_run/selected_inhomog_removal_models'

# mapping of Form names to actual names...
# These dicts are copied from another file...
selected_patient_files = {'7TMRI002': "7TMRI002",
                            '7TMRI003': "7TMRI003",
                            '7TMRI010': "7TMRI010",
                            '7TMRI015': "7TMRI015"}
# Reverse the relation..
selected_patient_files = {v: k for k, v in selected_patient_files.items()}

selected_volunteer_files = {'pr_06012021_1647041_12_3_t2wV4': "7TMRI100",
                            'v9_03032021_1641286_9_3_t2wV4': "7TMRI101",
                            'v9_08072020_1827193_9_3_t2wV4': "7TMRI102",
                            'v9_10022021_1720565_11_3_t2wV4': "7TMRI103",
                            'v9_11022021_1635487_6_3_t2wV4': "7TMRI104",
                            'v9_18012021_0927536_6_3_t2wV4': "7TMRI105",
                            'v9_27052020_1611487_22_3_t2wV4': "7TMRI106"}

# Reverse the relation..
selected_volunteer_files = {v: k for k, v in selected_volunteer_files.items()}
selected_patient_files.update(selected_volunteer_files)


fontsize = 16
font = {'family': 'normal',
        'weight': 'bold',
        'color': 'white',
        'size': fontsize}


contrast_answers = {1: "Klinisch niet bruikbaar",
2: "Contrast verslechterd, mogelijk gevolgen",
3: "Contrast verslechterd, niet klinisch relevant",
4: "Contrast niet verandert",
5: "Contrast verbeterd"}

homogeneity_answers = {1: "Homogeniteit verslechterd",
2: "Homogeniteit hetzelfde",
3: "Homogeniteit verbeterd, klinisch niet relevant",
4: "Homogeniteit verbeterd, prettiger in gebruik",
5: "Homogeniteit verbeterd"}

for sel_model_name, sel_model_dict in [('single_homogeneous', list_model_H_dict), ('single_biasfield', list_model_B_dict)]:
    fig, ax = plt.subplots(1, 3)
    fig.suptitle(sel_model_name)
    ax = ax.ravel()
    for ii, row_dict in enumerate(sel_model_dict):
        question_values = list(row_dict.keys())
        answer_values = list(row_dict.values())
        import matplotlib.pyplot as plt
        unique_questions = [x[11:] for x in question_values[:5]]
        unique_patient_id = [x[:8] for x in question_values[::5]]
        n_questions = len(unique_questions)
        for i, x in enumerate(unique_questions[:-2]):
            x_words = x.split(' ')
            n_words = len(x_words)
            title_string = ' '.join(x_words[:n_words // 2] + ['\n'] + x_words[n_words // 2:])
            nmax = 5
            waardering_array = [int(x) for x in answer_values[i::n_questions]]
            if 'contrast' in x:
                too_low_index = np.array(waardering_array) < 3
            else:
                too_low_index = np.array(waardering_array) < 2
            if any(too_low_index):
                print(sel_model_name, ii, x, too_low_index.sum())
                waardering_value = np.array(waardering_array)[too_low_index]
                bad_image_id = np.array(unique_patient_id)[too_low_index]
                bad_image_name = [selected_patient_files[x] for x in bad_image_id]
                bad_image_path = [
                    os.path.join(dmodel, sel_model_name, 'volunteer_corrected/pred_PNG', x + '.png') if x.endswith(
                        't2wV4')
                    else os.path.join(dmodel, sel_model_name, 'patient_corrected/pred_PNG', x + '.png') for x in
                    bad_image_name]
                for jj, sel_bad_image in enumerate(bad_image_path):
                    question_answer_value = waardering_value[jj]
                    if 'contrast' in x:
                        text_answer = contrast_answers[question_answer_value]
                    else:
                        text_answer = homogeneity_answers[question_answer_value]
                    model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(sel_bad_image))))
                    base_name = os.path.basename(sel_bad_image)
                    import helper.misc as hmisc
                    import helper.array_transf as harray
                    selected_image = hmisc.load_array(sel_bad_image, convert2gray=True)[:, :, 0]
                    input_image = hmisc.load_array(re.sub('pred_PNG', 'input_PNG', sel_bad_image), convert2gray=True)[:, :, 0]
                    center_mask = harray.create_random_center_mask(input_image.shape, mask_fraction=0.2)
                    cropped_selected_image, _ = harray.get_crop(selected_image, center_mask)
                    cropped_input_image, _ = harray.get_crop(input_image, center_mask)
                    # print(selected_image.shape)
                    fig_obj = hplotc.ListPlot([[input_image, cropped_input_image, selected_image, cropped_selected_image]], ax_off=True, title=x + f'\nAntwoord: {text_answer}',
                              sub_col_row=(2,2),
                    subtitle=[['Uncorrected image', 'Uncorrected image close up', 'Corrected image', 'Corrected image close up']], vmin=[(0, 0.5*input_image.max()), (0, 0.3*selected_image.max()), (0, 0.5*selected_image.max()), (0, 0.3*selected_image.max())])
                    # fig_obj.ax_list[0].text(10, fontsize, f'Antwoord: {text_answer}', **font)
                    print(os.path.join(dimg, base_name))
                    fig_obj.figure.savefig(os.path.join(dimg, str(ii) + "_question_" + str(i) + "_" +model_name + "_" + base_name))
                hplotc.close_all()



# Convert to GIF
import os
import helper.misc as hmisc

ddata = '/home/seb/data/ADNI/Selection_N4_RPI_one_per_visit_MNI'
ddest = '/home/seb/data/ADNI/GIF_version'

latest_file = hmisc.get_latest_file(ddata)

for ii, i_file in enumerate(os.listdir(ddata)):
    base_name = hmisc.get_base_name(i_file)
    sel_file = os.path.join(ddata, i_file)
    loaded_array = hmisc.load_array(sel_file).T[:, ::-1, ::-1]
    # loaded_array = np.moveaxis(loaded_array, 1, 0)
    n_card = loaded_array.shape[0]
    print(loaded_array.shape)
    hmisc.convert_image_to_gif(loaded_array,
                         output_path=os.path.join(ddest, f'{base_name}.gif'),
                         n_slices=n_card,
                         nx=128*1.2, ny=128,
                               duration=3./n_card)

i_file = hmisc.get_latest_file(ddata, n=2)
base_name = hmisc.get_base_name(i_file)
sel_file = os.path.join(ddata, i_file)
loaded_array = hmisc.load_array(sel_file).T[:, ::-1, ::-1]
# loaded_array = np.moveaxis(loaded_array, 1, 0)
n_card = loaded_array.shape[0]
print(loaded_array.shape)
hmisc.convert_image_to_gif(loaded_array,
                     output_path=os.path.join(ddest, f'{base_name}.gif'),
                     n_card=n_card,
                     nx=128*1.2, ny=128,
                           duration=3./n_card)
