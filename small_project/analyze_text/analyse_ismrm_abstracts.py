
import os
import numpy as np
import sys

with open('ismrm_reconstruction_abstracts.txt', 'r') as f:
    text_lines = f.readlines()

text_view, text_id, text_title, text_author = zip(*[x.split('\t') for x in text_lines])

import itertools
import collections
all_titles = ' '.join(text_title)

print('word', 'count')
for k, v in collections.Counter(all_titles.split(' ')).most_common(10):
    print(k, v)

# Search for specific title
sel_id = text_title.index('ODE-based Deep Network for MRI Reconstruction')
text_id[sel_id]

# Search for parallel stuff
[x for x in text_title if 'allel' in x]

# Search for specific title
sel_id = text_title.index('Plug-and-Play Deep Learning Module for Faster Parallel MR Imaging')
text_id[sel_id]

