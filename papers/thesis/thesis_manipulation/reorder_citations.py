import os
import numpy as np
import re
import collections
import pybtex
import itertools
import pybtex.database
from pybtex.utils import OrderedCaseInsensitiveDict
import helper.misc as hmisc

"""
The package I am ussing cannot deal properly with unsorted situations (in the bib file)
Hence, if I want to have citations that are numbered by occurence, then I need to re-order the citations

The logic will be as followed:

- load in a .tex file
- load in a .bib file

- extract all the cite commands from the tex files
- create a dict from the bib file (key - citation, value - content) -- using Pybtex for this
- create empty list for the new order
- while loop over the found cite keys
    - append to list if it is not in there
- re-create bib-file

"""


class ReOrderBib:
    def __init__(self, bib_file_name, tex_file_name, ddir):
        self.bib_file_name = bib_file_name
        self.tex_file_name = tex_file_name
        self.dest_directory = ddir

        self.bib_file = os.path.join(self.dest_directory, self.bib_file_name)
        self.tex_file = os.path.join(self.dest_directory, self.tex_file_name)

        # Used to get the cite names (non greedy)
        self.re_cite = re.compile("\\\cite\{(.*?)\}")

        self.text_str = self.read_tex_file()
        self.sorted_citation_keys = self.get_citations()
        self.bib_obj = self.read_bib_file()

    @staticmethod
    def create_backup_file(file_path):
        ext = hmisc.get_ext(file_path)
        basename = hmisc.get_base_name(file_path)
        dirname = os.path.dirname(file_path)
        backup_file = os.path.join(dirname, basename + "_backup" + ext)
        os.system(f'cp {file_path} {backup_file}')
        print(f'Created backup file {backup_file}')

    def read_tex_file(self):
        # Read the tex file
        with open(self.tex_file, 'r') as f:
            tex_text = f.read()
        return tex_text

    def read_bib_file(self):
        # Read the bib file
        return pybtex.database.parse_file(self.bib_file, bib_format='bibtex')

    def get_citations(self):
        # Get the sorted citations
        citation_keys = self.re_cite.findall(self.text_str)
        citation_keys = [x.strip() for x in itertools.chain(*[x.split(',') for x in citation_keys])]
        sorted_citation_keys = []
        for i_cite in citation_keys:
            if i_cite not in sorted_citation_keys:
                sorted_citation_keys.append(i_cite)
        return sorted_citation_keys

    def reorder_bib_obj(self):
        # Re-order the bib object
        ordered = OrderedCaseInsensitiveDict()
        for k in self.sorted_citation_keys:
            retrieved_citation = self.bib_obj.entries.get(k, None)
            if retrieved_citation:
                ordered[k] = retrieved_citation

        self.bib_obj.entries = ordered
        return self.bib_obj

    def write_bib_obj(self, bib_obj):
        # Create a backup of the original file
        self.create_backup_file(self.bib_file)
        # Overwrite the file
        bib_obj.to_file(self.bib_file)


if __name__ == "__main__":
    main_dir_name = '/home/bugger/Documents/paper/thesis/Thesis/Chapters'
    main_dir_name = '/home/bugger/Documents/paper/thesis/Thesis/FrontBackMatter/'
    # tex_file_name = 'main.tex'
    tex_file_name = 'Summary.tex'
    for i_chapter in os.listdir(main_dir_name):
        dchapter = os.path.join(main_dir_name, i_chapter)
        if os.path.isdir(dchapter):
            bib_file_list = [x for x in os.listdir(dchapter) if x.endswith('bib')]
            if len(bib_file_list):
                bib_file_name = bib_file_list[0]
                print(dchapter, bib_file_name, tex_file_name)
                reorder_bib = ReOrderBib(bib_file_name=bib_file_name, tex_file_name=tex_file_name, ddir=dchapter)
                reordered_bib_obj = reorder_bib.reorder_bib_obj()
                reorder_bib.write_bib_obj(reordered_bib_obj)