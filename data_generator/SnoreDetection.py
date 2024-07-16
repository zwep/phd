import re
import numpy as np
import os
import data_generator.Generic as data_gen
import torch
import torchaudio
import torchaudio.transforms
import re


class DataGeneratorFullBodyProstate(data_gen.DatasetGenericComplex):
    """
    Simple Audio generator

    Could add the nfft I guess
    """

    def __getitem__(self, index):
        sel_dataset = np.random.choice(self.n_datasets)
        file_list = self.container_file_info[sel_dataset]['file_list']
        input_dir = self.container_file_info[sel_dataset]['input_dir']
        target_dir = self.container_file_info[sel_dataset]['target_dir']
        mask_dir = self.container_file_info[sel_dataset]['mask_dir']
        index = index % len(file_list) # Make sure that
        i_file = file_list[index]

        target_indicator = re.findall("^([0-9])_", i_file)
        if target_indicator:
            target_value = int(target_indicator[0])
        else:
            target_value = -1
            print('Regex failed.. string is', i_file)

        target_tensor = torch.Tensor(target_value).int()

        input_file = os.path.join(input_dir, i_file)
        input_tensor = torchaudio.load_wav(input_file)
        input_spectro = torchaudio.transforms.Spectrogram(input_tensor)
        print('Size of the spectro thing ', input_spectro)


        container = {'input': input_spectro, 'target': target_tensor}
        return container