"""
This is not great...

But it is something
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2 # Difference between data1 and data2
    md = np.mean(diff) # Mean of the difference
    sd = np.std(diff, axis=0) # Standard deviation of the difference
    fig, ax = plt.subplots()
    ax.scatter(mean, diff, *args, **kwargs)
    ax.axhline(md,           color='gray', linestyle='--')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax.set_ylim(md - 2 * 1.96*sd, md + 2 * 1.96*sd)


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


# Get the answers per reviewer..
reviewer_answers = [[], []]
reviewer_answers_binary = [[], []]
reviewer_answers_model = {'Homogeneous network': [],
                          'Biasfield network': []}
reviewer_answers_model_homog = {'Homogeneous network': [],
                                'Biasfield network': []}
reviewer_answers_model_cntrst = {'Homogeneous network': [],
                                'Biasfield network': []}
for sel_model_name, sel_model_dict in [('Homogeneous network', list_model_H_dict), ('Biasfield network', list_model_B_dict)]:
    for ii_reviewer, row_dict in enumerate(sel_model_dict):
        question_values = list(row_dict.keys())
        answer_values = list(row_dict.values())
        import matplotlib.pyplot as plt
        unique_questions = [x[11:] for x in question_values[:5]]
        n_questions = len(unique_questions)
        for i, x in enumerate(unique_questions[:-2]):
            waardering_array = [int(x) for x in answer_values[i::n_questions]]
            reviewer_answers[ii_reviewer].extend(waardering_array)
            if 'contrast' in x:
                good_or_bad = np.array(waardering_array) > 3
                reviewer_answers_model_cntrst[sel_model_name].extend(waardering_array)
            else:
                good_or_bad = np.array(waardering_array) > 1
                reviewer_answers_model_homog[sel_model_name].extend(waardering_array)
            reviewer_answers_binary[ii_reviewer].extend(list(good_or_bad))
            reviewer_answers_model[sel_model_name].extend(waardering_array)

data_1 = np.array(reviewer_answers[0])
data_2 = np.array(reviewer_answers[1])

bland_altman_plot(data_1, data_2)

(data_2 == data_1).sum() / len(data_2)
(data_2 == data_1).sum() / len(data_2)

data_1 = np.array(reviewer_answers_binary[0])
np.mean(data_1)
data_2 = np.array(reviewer_answers_binary[1])
np.mean(data_2)

bland_altman_plot(data_1.astype(int), data_2.astype(int))

(data_2 == data_1).sum() / len(data_2)
(data_2 == data_1).sum() / len(data_2)

for k, v in reviewer_answers_model.items():
    print(k, np.mean(v))

for k, v in reviewer_answers_model_cntrst.items():
    print(k, np.mean(v))

for k, v in reviewer_answers_model_homog.items():
    print(k, np.mean(v))