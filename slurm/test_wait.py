import sys
import time
import torch
import numpy as np
import helper.nvidia_parser as hnvidia

gpu_id, _ = hnvidia.get_free_gpu_id()
A = np.random.rand(100, 100, 100)
A_tens = torch.from_numpy(A)
device = torch.device("cuda:{}".format(str(gpu_id)) if torch.cuda.is_available() else "cpu")
A_tens.to(device)

print('Sleeping for 5 minutes')
time.sleep(300)
print('Done sleeping')
