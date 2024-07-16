import os
import csv
import numpy as np
import matplotlib.pyplot as plt

dcsv = '/home/bugger/Documents/paper/inhomogeneity removal/response_google_forms/bertine_joost_Review prostaat beelden.csv'

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

title_font_size = 20
question_font_size = 16
ticks_font_size = 16
dutch_to_english = {
'Verbetering of behoud van het contrast in de prostaat?': 'Improvement or conservation of the contrast in the prostate?',
'Verbetering of behoud van de homogeniteit in de prostaat en omliggend weefsel?': 'Improvement or conservation of the homogeneity in the prostate and surrounding tissue?',
'Verbetering of behoud van de homogeniteit in het hele lichaam?': 'Improvement or conservation of the homogeneity in the whole body?',
'Bruikbaarheid van het corrigeerde beeld': 'Usefulness of the corrected image',
'Bruikbaarheid van het oncorrigeerde beeld': 'Usefulness of the unccorrected image'}

for sel_model_name, sel_model_dict in [('Homogeneous network', list_model_H_dict), ('Biasfield network', list_model_B_dict)]:
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    # fig.suptitle(sel_model_name, fontsize=title_font_size)
    ax = ax.ravel()
    for iax in ax:
        iax.tick_params(axis='y', labelrotation=0, labelsize=ticks_font_size)
        iax.tick_params(axis='x', labelrotation=0, labelsize=ticks_font_size)
    for ii, row_dict in enumerate(sel_model_dict):
        question_values = list(row_dict.keys())
        answer_values = list(row_dict.values())
        import matplotlib.pyplot as plt
        unique_questions = [x[11:] for x in question_values[:5]]
        n_questions = len(unique_questions)
        for i, x in enumerate(unique_questions[:-2]):
            x = dutch_to_english[x]
            print(i)
            x_words = x.split(' ')
            n_words = len(x_words)
            title_string = ' '.join(x_words[:n_words // 2] + ['\n'] + x_words[n_words // 2:])
            nmax = 5
            waardering_array = [int(x) for x in answer_values[i::n_questions]]
            print(i, waardering_array)
            ax[i].set_title(title_string, fontsize=question_font_size)
            ax[i].hist(waardering_array, np.arange(1, nmax+2)-0.5, density=False, alpha=0.5, label=f'Reviewer {str(ii)}')
            ax[i].set_ylim(0, 12)
            ax[i].legend()
            if i == 0:
                ax[i].vlines(x=3.5, ymin=0, ymax=12, colors='red', linestyles='--')
            elif i > 0:
                ax[i].vlines(x=1.5, ymin=0, ymax=12, colors='red', linestyles='--')

        ax[0].set_ylabel('Number of cases', fontsize=title_font_size)
        fig.supxlabel('Score', fontsize=title_font_size)
        # fig.supylabel('Number of cases')
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.pause(0.1)
    fig.savefig(f'/home/bugger/Documents/paper/inhomogeneity removal/response_google_forms/{sel_model_name}.png', bbox_inches='tight', pad_inches=0.0)
    # This code showed the usefulness of the images..
    # title_string = 'Bruikbaarheid van het beeld'
    # ax[-1].set_title(title_string)
    # nmax = 10
    # # Question corrected
    # waardering_array = [int(x) for x in answer_values[3::n_questions]]
    # ax[-1].hist(waardering_array, np.arange(1, nmax+2)-0.5, density=False, color='red', alpha=0.5, label='corrected')
    # # Question uncorrected
    # waardering_array = [int(x) for x in answer_values[4::n_questions]]
    # ax[-1].set_title(title_string)
    # ax[-1].hist(waardering_array, np.arange(1, nmax+2)-0.5, density=False, color='blue', alpha=0.5, label='uncorrected')
    # ax[-1].legend()
    # fig.suptitle(model_name)
    # plt.tight_layout()
