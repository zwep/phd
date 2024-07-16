import h5py
import os
import numpy as np
import time
import timeit

t0 = time.time()

def load_fun():
    h5_file = os.path.expanduser('~/v9_21102020_1703193_10_3_transradialfastV4.h5')

    with h5py.File(h5_file, 'r') as f:
        A = np.array(f['kspace'])

load_fun()
print('Elapsed time by timeit', time.time() - t0)


testcode= '''
def load_fun():
    h5_file = os.path.expanduser('~/v9_21102020_1703193_10_3_transradialfastV4.h5')

    with h5py.File(h5_file, 'r') as f:
        A = np.array(f['kspace'])
'''
import_module="import numpy, h5py"
print(np.mean(timeit.repeat(stmt=testcode, setup=import_module, repeat=100)))