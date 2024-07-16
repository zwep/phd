import matplotlib.pyplot as plt
import numpy as np
import collections
import itertools
import os


dthesis = '/home/bugger/Documents/paper/thesis/Thesis/Chapters/'

collection_tex = ''
for d, _, f in os.walk(dthesis):
    filter_f = [x for x in f if x.endswith('main.tex')]
    if len(filter_f):
        tex_file = os.path.join(d, filter_f[0])
        with open(tex_file, 'r') as tex_file_obj:
            tex_content = tex_file_obj.read()
        collection_tex += tex_content


#collection_tex = list(itertools.chain(*collection_tex))
counter_obj = collections.Counter(collection_tex.split())
top_100 = counter_obj.most_common(100)
top_100_keys, top_100_values = zip(*top_100)
top_100_values = np.array(top_100_values) / max(top_100_values)

fig, ax = plt.subplots()
ax.plot(top_100_values, label='word_count')
ax.plot(1 / np.arange(1, 101), label="Zipf's law")
plt.legend()