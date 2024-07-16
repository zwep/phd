# Stupid bibtex

import os
import re


# def correct_bib_title(bib_text):
#     for ii in range(len(bib_text)):
#         i_line = bib_text[ii]
#         if i_line.strip().startswith("title"):
#
#     return bib_text

ddata = "/home/bugger/Documents/paper/thesis/Thesis/Chapters/"
# ddata = "/home/bugger/Documents/paper/thesis/Thesis/FrontBackMatter/"
chapter_dir = os.listdir(ddata)

chapter_index = 0
for chapter_index in range(len(chapter_dir)):
    chapter_path = os.path.join(ddata, chapter_dir[chapter_index])
    if os.path.isdir(chapter_path):
        bib_file_name = [x for x in os.listdir(chapter_path) if x.endswith("bib")]
        if len(bib_file_name):
            bib_file_name = bib_file_name[0]
            bib_path = os.path.join(chapter_path, bib_file_name)
            # bib_path = '/home/bugger/Documents/paper/thesis/Thesis/Chapters/Introduction/introduction.bib'
            print('\n\nOpen ', bib_path)
            if os.path.isfile(bib_path):
                with open(bib_path, "r") as f:
                    bib_file = f.readlines()

                for ii in range(len(bib_file)):
                    i_line = bib_file[ii]
                    if i_line.strip().startswith("title"):
                        _, title_str = i_line.strip().split('=')
                        title_str = re.sub('\{|\}', '', title_str).strip()
                        n_capital_words = sum(1 for x in title_str.split(' ') if x[0].isupper())
                        if n_capital_words > 2:
                            print(title_str, n_capital_words)
