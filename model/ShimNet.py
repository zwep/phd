"""
Model we use for Shimming stuff.
"""

import torch
import torch.nn as nn
import model.Blocks as Blocks
import numpy as np
import helper_torch.activations as htactv


class ShimNet(nn.Module):
    """
    Random stuff for shimming....
    """

    def __init__(self, in_chan, out_chan, n_downsample, input_shape,  **kwargs):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_downsample = n_downsample

        self.debug = kwargs.get('debug', False)
        channel_size = in_chan * 2 ** n_downsample
        image_size = np.array(input_shape) / (2 ** (2 * n_downsample))
        image_size = image_size.astype(int)
        self.dense_layer_size = np.prod(image_size) * channel_size

        self.model_layers = self.get_model()

    def get_model(self):
        layers = []
        ch = self.in_chan
        for i_layer in range(self.n_downsample):
            temp_down = Blocks.ConvBlock2D(in_chans=ch, out_chans=ch)
            temp_sample = nn.Conv2d(in_channels=ch, out_channels=2*ch, kernel_size=4, stride=4, groups=1)
            layers.append(temp_down)
            layers.append(temp_sample)
            ch = 2 * ch

        flatten_layer = nn.Flatten()
        dense_layer = nn.Linear(in_features=self.dense_layer_size, out_features=self.out_chan)
        # tanh_act = htactv.ScaledTanh()
        layers.append(flatten_layer)
        layers.append(dense_layer)
        # layers.append(tanh_act)
        model_layers = nn.ModuleList(layers)

        return model_layers

    def forward(self, x):
        for i_layer in self.model_layers:
            if self.debug:
                print(x.shape)
            x = i_layer(x)
        return x


class PienNet(nn.Module):
    def __init__(self, dense_layer_size, debug=False):
        super().__init__()
        self.debug = debug
        self.dense_layer_size = dense_layer_size
        self.model_layers = self.get_model()

    def get_model(self):

        conv1 = nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        actv1 = nn.ReLU()
        max1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2))
        actv2 = nn.ReLU()
        max2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        actv3 = nn.ReLU()
        max3 = nn.MaxPool2d(kernel_size=(2, 2))

        conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        actv4 = nn.ReLU()
        max4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        flatten_layer = nn.Flatten()
        dense_layer = nn.Linear(in_features=self.dense_layer_size, out_features=16)

        layers = [conv1, actv1, max1, conv2, actv2, max2, conv3, actv3, max3, conv4, actv4, max4, flatten_layer, dense_layer]
        model_layers = nn.ModuleList(layers)
        return model_layers

    def forward(self, x):
        for i_layer in self.model_layers:
            if self.debug:
                print(x.shape)
            x = i_layer(x)
        return x


if __name__ == "__main__":
    import numpy as np
    input_shape = (243, 205)
    A = np.random.rand(1, 128, 128, 128)
    A_tens = torch.from_numpy(A).float()
    model_obj = ShimNet(in_chan=128, out_chan=16, n_downsample=3, input_shape=input_shape, debug=True)
    res = model_obj(A_tens)

    model_obj = PienNet(dense_layer_size=1152, debug=True)
    model_obj(A_tens)