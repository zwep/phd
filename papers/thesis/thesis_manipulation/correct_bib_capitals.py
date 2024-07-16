# Stupid bibtex

import os
import re


def correct_bib_title(bib_text):
    for ii in range(len(bib_text)):
        i_line = bib_text[ii]
        if i_line.strip().startswith("title"):
            i_line = re.sub("(\{+)", "{", i_line)
            i_line = re.sub("(\}+)", "}", i_line)
            i_line = re.sub("\{", "{{", i_line)
            i_line = re.sub("\}", "}}", i_line)
            bib_text[ii] = i_line

    return bib_text

ddata = "/home/bugger/Documents/paper/thesis/Thesis/Chapters/"
ddata = "/home/bugger/Documents/paper/thesis/Thesis/FrontBackMatter/"
chapter_dir = os.listdir(ddata)

chapter_index = 0
for chapter_index in range(len(chapter_dir)):
    chapter_path = os.path.join(ddata, chapter_dir[chapter_index])
    if os.path.isdir(chapter_path):
        bib_file_name = [x for x in os.listdir(chapter_path) if x.endswith("bib")]
        if len(bib_file_name):
            bib_file_name = bib_file_name[0]
            bib_path = os.path.join(chapter_path, bib_file_name)
            if os.path.isfile(bib_path):
                with open(bib_path, "r") as f:
                    bib_file = f.readlines()

                bib_file = correct_bib_title(bib_file)

                with open(bib_path, "w") as f:
                    f.write("".join(bib_file))