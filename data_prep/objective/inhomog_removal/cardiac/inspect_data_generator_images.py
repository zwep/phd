"""
Check if we create images are OK or not
"""

import numpy as np
import helper.plot_class as hplotc
import os
import data_generator.InhomogRemoval as data_gen

# 4ch/2ch and transverse
ddata = '/data/seb/semireal/cardiac_simulation_rxtx/p2ch_registered;/data/seb/semireal/cardiac_simulation_rxtx/p4ch_registered'

data_gen_obj = data_gen.DataGeneratorCardiac(ddata=ddata, target_type='biasfield', transform_type_target='abs')
cont = data_gen_obj.__getitem__(0)
inp_np = cont['input'].numpy()
tgt_np = cont['target'].numpy()
plot_obj = hplotc.ListPlot([inp_np, tgt_np], cbar=True)
plot_obj.figure.savefig('/data/seb/check_data_gen.png')
