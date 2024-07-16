import torch
import numpy as np
from data_generator.Generic import DatasetGenericComplex, DatasetGeneric
import os


class DataGeneratorPhysicsInspired(DatasetGeneric):
    """
    Simple data generator... for now.. gives Noise as input. Target does nothing.
    """
    def __init__(self, ddata, input_shape, target_shape=None,
                 shuffle=True, dataset_type='train', file_ext='npy', **kwargs):

        super().__init__(ddata, input_shape, target_shape=target_shape, shuffle=shuffle,
                         dataset_type=dataset_type, file_ext=file_ext, **kwargs)

        self.complex_output = kwargs.get('complex_output', False)

    def __getitem__(self, item):
        # NOw it is between -1..1
        # input_array = 2 * np.random.random(self.img_input_shape) - 1
        input_array = np.zeros(self.img_input_shape)
        # First boundary condition
        # b0 = np.exp(1j * np.random.uniform(0, 2 * np.pi))
        b0 = np.random.random() * 2 - 1
        # Second boundary condition
        # b1 = np.exp(1j * np.random.uniform(0, 2 * np.pi))
        b1 = np.random.random() * 2 - 1
        input_array[0, 0:10] = np.real(b0)
        input_array[0, -10:] = np.real(b1)
        if self.complex_output:
            input_array[1, 0:10] = np.imag(b0)
            input_array[1, -10:] = np.imag(b1)

        # This should... have the boundary conditions I guess...
        # We have constructed a VERY specific loss. It extracts the boundary conditions.. and
        # uses the 0-vector elsewhere
        target_array = np.zeros(self.img_target_shape)
        target_array[0, 0:10] = np.real(b0)
        target_array[0, -10:] = np.real(b1)
        if self.complex_output:
            target_array[1, 0:10] = np.imag(b0)
            target_array[1, -10:] = np.imag(b1)

        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(target_array).float()

        return {'input': input_tensor, 'target': target_tensor}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    A = DataGeneratorPhysicsInspired('', (2, 100), number_of_examples=100)
    container = A.__getitem__(0)
    plt.plot(container['input'].numpy()[0])
    plt.plot(container['input'].numpy()[1])

    import helper_torch.loss as hloss
    A_loss = hloss.HelmholtzLoss1D()
    A_loss(container['input'][None], container['target'][None])