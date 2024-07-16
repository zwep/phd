# encoding: utf-8

"""
Here we define all the necessary functions to choose a GPU device while using Keras Backend (Tensorflow)


##Example of usage Keras (before tensorflow 2.0)

index_gpu, p_gpu = get_free_gpu_id(claim_memory=0.99)
sess = get_gpu_session(index_gpu, p_gpu)
K.tensorflow_backend.set_session(sess)


##Example of usage torch

index_gpu, p_gpu = get_free_gpu_id(claim_memory=0.99)
device = torch.device("cuda:{}".format(str(index_gpu)) if torch.cuda.is_available() else "cpu")
model.to(device)
"""

import subprocess
import re


def parse_free_memory(output):
    """
     $ nvidia-smi -q -d MEMORY -i 1
    ==============NVSMI LOG==============
    Timestamp                           : Wed Jan 31 14:13:22 2018
    Driver Version                      : 384.111
    Attached GPUs                       : 2
    GPU 00000000:06:00.0
        FB Memory Usage
            Total                       : 11172 MiB
            Used                        : 10 MiB
            Free                        : 11162 MiB
        BAR1 Memory Usage
            Total                       : 256 MiB
            Used                        : 2 MiB
            Free                        : 254 MiB

    :param output: String from nvidia-smi -q -d output
    :return: Free memory (in MiB) and total memory (in MiB)
    """

    pat_free = re.compile(r"Free\s+:\s+(\d+)\s+MiB")
    pat_total = re.compile(r"Total\s+:\s+(\d+)\s+MiB")

    m_free = int(pat_free.search(output).group(1))
    m_total = int(pat_total.search(output).group(1))

    return m_free, m_total


def get_num_gpu():
    """
    We can also use from tensorflow.python.client
        import device_lib
        device_lib.device_list()

    However this will allocate some memory on the GPUs themselves to check for something. So we'll use nvidia-smi -L

    :return: Number of GPUs by parsing nvidia-smi
    """
    try:
        res = subprocess.check_output("nvidia-smi -L", shell=True).decode('utf-8').split('\n')
        res = [x for x in res if len(x) > 0]
    except subprocess.CalledProcessError:
        print('NVIDIA is not installed')
        res = []
    except OSError:
        print('NVIDIA is not installed')
        res = []
    return len(res)


def get_free_memory_gpu():
    """
    nvidia-smi -q -d MEMORY -i

    :param claim_memory: either a fraction (of the whole gpu mem) or a set amount of MiB that you wish to occupy.
    :return: index of gpu and percentage.
    """

    # Used when not viable GPU was found under current settings
    ngpus = get_num_gpu()

    if ngpus:
        avail = [0] * ngpus
        total_avail = [0] * ngpus
        nvidia_cmd = ["nvidia-smi", "-q", "-d", "MEMORY", "-i"]

        # Loop over all available GPUs
        for i in range(ngpus):
            cmd = nvidia_cmd + [str(i)]
            output = subprocess.check_output(cmd).decode("utf-8")
            # Extract memory usage value
            avail[i], total_avail[i] = parse_free_memory(output)
        return avail, total_avail
    else:
        return [], []


def get_free_gpu_id(claim_memory=0.0):
    """
    nvidia-smi -q -d MEMORY -i

    :param claim_memory: either a fraction (of the whole gpu mem) or a set amount of MiB that you wish to occupy.
    :return: index of gpu and percentage.
    """

    # Used when not viable GPU was found under current settings
    safety_factor = 0.9
    ngpus = get_num_gpu()

    if ngpus:
        avail = [0] * ngpus
        total_avail = [0] * ngpus
        nvidia_cmd = ["nvidia-smi", "-q", "-d", "MEMORY", "-i"]

        # Loop over all available GPUs
        for i in range(ngpus):
            cmd = nvidia_cmd + [str(i)]
            output = subprocess.check_output(cmd).decode("utf-8")
            # Extract memory usage value
            avail[i], total_avail[i] = parse_free_memory(output)

        # Calculate usage percentage
        percent_avail = [x / y for x, y in zip(avail, total_avail)]

        # Set the right array from which we want to filter, based on claim_memory
        # (Differentiate between percentage and absolute memory
        if claim_memory < 1:
            avail_mem = percent_avail
        else:
            avail_mem = avail

        # Make only those GPUs visible that satisfy the memory constraint imposed by claim_memory
        avail_mem_filter = [x if x >= claim_memory else 0 for x in avail_mem]

        # If nothing satisfies the condition, propose an alternative
        if all([x == 0 for x in avail_mem_filter]):
            gpu_ind = -1
            p_claim_memory = 0
            print('No viable GPU was found. Setting selection to CPU.')

            # Possibility to offer a choice of a new GPU
            # gpu_ind = -1
            # avail_mem_value = max(percent_avail)
            # alt_gpu_ind = percent_avail.index(avail_mem_value)
            # total_avail_value = total_avail[alt_gpu_ind]
            # p_claim_memory = {'alt_index_gpu': alt_gpu_ind,
            #                   'alt_avail_mem_perc': avail_mem_value,
            #                   'alt_avail_mem_mib': avail_mem_value*total_avail_value}

        else:
            avail_mem_value = max(avail_mem_filter)
            gpu_ind = avail_mem_filter.index(avail_mem_value)
            total_avail_value = total_avail[gpu_ind]

            # Translate the requested memory (in MiB) into a fraction
            if claim_memory < 1:
                p_claim_memory = claim_memory
            else:
                p_claim_memory = claim_memory/total_avail_value
    else:
        print('No GPUs found')
        gpu_ind = None
        p_claim_memory = 0

    return gpu_ind, p_claim_memory


if __name__ == "__main__":
    import os
    gpu_ind, p_claim_memory = get_free_gpu_id(0.8)
    os.system(f'echo {gpu_ind}')
