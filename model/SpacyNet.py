import torch.nn.functional as F
import torch
import torch.nn as nn
import helper.spacy as hspacy
import numpy as np


class SpacyNet(nn.Module):
    def __init__(self, n_degree, in_chan, **kwargs):
        super().__init__()
        n_x = 128
        n_y = 128
        spacy_matrix = hspacy.get_spacy_matrix(n_x=2 * n_x, n_y=2 * n_y, n_degree=n_degree)
        n_classes = spacy_matrix.shape[-1]
        self.in_chan = in_chan
        self.n_x = n_x
        self.n_y = n_y
        coil_position = self.get_coil_position()

        temp_spacy_reshp = spacy_matrix.reshape((2 * n_y, 2 * n_y, n_classes))

        temp_spacy_list = []
        for sel_x, sel_y in coil_position:
            sel_x = sel_x // 2
            sel_y = sel_y // 2
            # print(n_size - sel_x, 2 * n_size - sel_x, n_size - sel_y, 2 * n_size - sel_y)
            temp_subset = temp_spacy_reshp[n_x - sel_x: 2 * n_x - sel_x, n_y - sel_y: 2 * n_y - sel_y]
            temp_spacy_list.append(temp_subset.reshape((-1, n_classes)))

        temp_spacy = np.concatenate(temp_spacy_list, axis=1)
        self.n_classes = temp_spacy.shape[1]
        spacy_matrix = torch.as_tensor(temp_spacy.real).float()
        self.spacy_matrix_T = nn.Parameter(torch.as_tensor(spacy_matrix.T), requires_grad=False)

        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.conv1_1 = nn.Conv2d(in_chan, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # max pooling (kernel_size, stride)
        self.pool = nn.MaxPool2d(2, 2)

        # fully conected layers:
        self.fc6 = nn.Linear(4*4*512, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, self.n_classes)

    @staticmethod
    def get_coil_position():
        # Average coil positoin calculated over many coils..
        # Obtained via `get_average_coil_position.py` script
        average_coil_pos = np.array(
            [[82.11470037, 239.8338015],
             [48.88810861, 169.99656679],
             [47.68789014, 93.52621723],
             [71.42993134, 20.5744382],
             [207.14325843, 16.53792135],
             [225.85549313, 86.63623596],
             [226.76186017, 161.39138577],
             [216.22487516, 235.51513733]]
        ).astype(int)
        return average_coil_pos

    def forward(self, x, training=True):
        x = F.interpolate(x, size=(128, 128), mode='bicubic')
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool(x)
        x = x.view(-1, 4 * 4 * 512)
        print(x.shape)
        x = F.relu(self.fc6(x))
        print(x.shape)
        x = F.dropout(x, 0.5, training=training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, training=training)
        x = self.fc8(x)
        print(x.shape, self.spacy_matrix_T.shape)
        x = torch.matmul(x, self.spacy_matrix_T)
        x = x.view(-1, self.n_y, self.n_x)
        return x


if __name__ == "__main__":
    import data_generator.InhomogRemoval as data_gen
    model_obj = SpacyNet(n_degree=8, n_x=256, n_y=256, in_chan=16)

    # Inhomogeneity removal the old-skool way on Gradient Echo data.
    dir_data = '/home/bugger/Documents/data/semireal/prostate_simulation_rxtx'
    gen = data_gen.DataGeneratorInhomogRemoval(ddata=dir_data, dataset_type='test', complex_type='cartesian',
                                      input_shape=(1, 256, 256),
                                      alternative_input='/home/bugger/Documents/data/celeba',
                                      use_tx_shim=False,
                                      b1m_scaling=False,
                                      b1p_scaling=True,
                                      debug=True,
                                      masked=True,
                                      target_type='b1p')

    container = gen.__getitem__(0)
    inp = container['input'].numpy()
    tgt = container['target'].numpy()

    import helper.plot_fun as hplotf
    import helper.plot_class as hplotc

    hplotc.SlidingPlot(inp)
    res = model_obj(container['input'][None])

    hplotf.plot_3d_list(res.detach().numpy().real)