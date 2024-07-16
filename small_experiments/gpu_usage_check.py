
"""
I have A LOT of memory usage on one of the GPUs

Need to see if everything goes well...
Start off with some simple calculations
"""

import sys
import torch
import numpy as np
import GPUtil
import gc

def check_gc_objects():
    tensor_obj = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.__sizeof__())
                tensor_obj.append(obj)
        except:
            pass
    return tensor_obj

print('Intitial status:')
check_gc_objects()

A = np.random.rand(1)
A_tensor = torch.from_numpy(A)
index_gpu = 2
gpu_device = torch.device("cuda:{}".format(str(index_gpu)))

n_objects = 10
# This get the current allocated bytes
for i in range(n_objects):
    print(i, '\t Bytes ', torch.cuda.memory_stats(device=gpu_device)['allocated_bytes.all.current'])
    # A = np.random.rand(1024, 1024)
    # A_tensor = torch.from_numpy(A)
    # A_tensor.to(gpu_device)
    torch.from_numpy(np.random.rand(1024, 1024, 1024)).to(gpu_device)

torch.cuda.ipc_collect()
torch.cuda.memory_stats(device=gpu_device)
torch.cuda.memory_summary(device=gpu_device)

GPUs = GPUtil.getGPUs()

(gpu.load*100, gpu.memoryUtil*100)
GPUtil.showUtilization()
torch.cuda.empty_cache()
