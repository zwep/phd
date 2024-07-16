

import torch.nn as nn
import torch
import numpy as np
import helper.plot_fun as hplotf


class AE(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, in_chan, out_chan, out_size=256, **kwargs):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.out_size = out_size
        self.layers = self.get_model()
        self.debug = kwargs.get('debug', False)

    def get_model(self):
        #encoder
        model_obj = nn.ModuleList([
                                 nn.Conv2d(self.in_chan, 12, kernel_size=7),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(12),

                                 nn.Conv2d(12, 12, (7, 7)),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(12),

                                 nn.MaxPool2d(kernel_size=2),

                                 nn.Conv2d(12, 24, kernel_size=7),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(24),

                                 nn.Conv2d(24, 24, kernel_size=7),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(24),

                                 nn.MaxPool2d(kernel_size=2),

                                 nn.Conv2d(24, 24, kernel_size=7),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(24),

                                 nn.Conv2d(24, 24, kernel_size=7),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(24),

                                 #decoder
                                 nn.ConvTranspose2d(24, 24, kernel_size=2, stride=2),

                                 nn.Conv2d(24, 12, kernel_size=7),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(12),

                                 nn.ConvTranspose2d(12, 12, kernel_size=2, stride=2),

                                 nn.Conv2d(12, 12, kernel_size=7),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(12),

                                 nn.ConvTranspose2d(12, 12, kernel_size=2, stride=2),

                                 nn.Conv2d(12, out_channels=self.out_chan, kernel_size=7)])

        return model_obj

    def forward(self, x):
        for i_layer in self.layers:
            if self.debug:
                print(x.shape)
            x = i_layer(x)
        x_shape, y_shape = x.shape[-2:]
        return x[:, :, x_shape//2 - self.out_size//2: x_shape//2 + self.out_size//2, y_shape//2 - self.out_size//2: y_shape//2 + self.out_size//2]


if __name__ == "__main__":
    model_obj = AE().float()
    A_in = torch.as_tensor(np.random.rand(256, 256)).float()
    res = model_obj(A_in[None, None])
    res.shape
