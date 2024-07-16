from helper.nvidia_parser import get_free_memory_gpu
import os
import time
import sys
import argparse

"""
People are using all the GPUs...

I cant sit by myself and wait for this

Continue on this later...?
"""

parser = argparse.ArgumentParser()
parser.add_argument('-script')
parser.add_argument('-mem')

p_args = parser.parse_args()

script_to_run = p_args.script
required_memory = int(p_args.mem)

looking_for_gpu = True
while looking_for_gpu:
    available, total = get_free_memory_gpu()
    print('\n\n===========')
    for ii, i_avail in enumerate(available):
        print(f'GPU {ii}: available memory {i_avail} Mb')

    check_memory = [x >= required_memory for x in available]
    if any(check_memory):
        os.system(script_to_run)
        looking_for_gpu = False
    else:
        time.sleep(60)