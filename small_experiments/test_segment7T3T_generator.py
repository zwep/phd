
import nibabel
# Check the size of the data....
import os
import numpy as np
import helper.plot_class as hplotc
import helper.misc as hmisc
import helper.array_transf as harray
import data_generator.Segment7T3T as data_gen
ddata = '/data/cmr7t3t/biasfield_sa_mm1_B'
gen_obj = data_gen.DataGeneratorCardiacSegment(ddata, target_type='segmentation',
                                      transform_type='abs', presentation_mode=True,
                                      dataset_type='test')
for i in range(10):
    container = gen_obj.__getitem__(i)
    print(gen_obj.container_file_info[0]['file_list'][i])
    inp = container['input']
    # fig_obj = hplotc.ListPlot(inp)
    # fig_obj.figure.savefig(f'/data/seb/input_{i}.png')
    fig_obj2 = hplotc.ListPlot([container['b1m_shim'], container['b1p_shim']], augm='np.abs')
    fig_obj2.figure.savefig(f'/data/seb/b1shim_{i}.png')
    A = np.abs(container['b1m_shim'])
    A1 = np.abs(container['b1p_signal'])
    B = (A > 1e-3).astype(int)
    B1 = (A1 > 1e-3).astype(int)
    C = harray.get_treshold_label_mask(A)
    C1 = harray.get_treshold_label_mask(A1)
    print('B1m', harray.get_minmeanmediammax(A))
    print('B1p', harray.get_minmeanmediammax(A1))
    fig_obj3 = hplotc.ListPlot([A, A1, B, B1, container['mask'], inp, container['target']])
    fig_obj3.figure.savefig(f'/data/seb/mask_{i}.png')