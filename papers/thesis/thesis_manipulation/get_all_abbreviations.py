
# We want to collect all the abbreviations in the thesis
# For me those are words consisting of two or more capital letters
# I want to keep a list per chapter
# And denote in which sentence (or line) this occurs

import os
import re


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_abbrv_dict(tex_file, re_comp=re.compile("(\s+|\()([A-Z]{2}[A-Z]*)(\.|,|\s+|-|\))")):
    # collect all the abbreviations
    abbreviation_dict = dict()
    for ii, i_line in enumerate(tex_file):
        i_line = i_line.strip()
        # Dont consider any comments or fuctions like \includegraphs...
        if i_line.startswith("%") or i_line.startswith("\\"):
            continue
        # Find all capitalized words...
        re_obj = re_comp.findall(i_line)
        if re_obj:
            for i_obj in re_obj:
                i_obj = i_obj[1]  # here we remove the optional space
                if i_obj in abbreviation_dict:
                    pass
                else:
                    abbreviation_dict.setdefault(i_obj, ii)
    return abbreviation_dict


ddata = "/home/bugger/Documents/paper/thesis/Thesis/Chapters/"
chapter_dir = os.listdir(ddata)


chapter_index = 4+2
main_tex_file = os.path.join(ddata, chapter_dir[chapter_index], "main.tex")

print(f" Reading {chapter_dir[chapter_index]} ")
with open(main_tex_file, "r") as f:
    main_tex_text = f.readlines()

abbreviation_dict = get_abbrv_dict(main_tex_text)

for k, v in abbreviation_dict.items():
    print()
    print()
    print(f"{bcolors.OKCYAN}{k}{bcolors.ENDC}".join(main_tex_text[v].split(k)))


