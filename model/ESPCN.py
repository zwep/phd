
import warnings
import torch.nn as nn
import math
import model.Blocks as mblocks
import helper_torch.layers as hlayers


class ESPCN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, drop_prob=0.0, **kwargs):
        super().__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=5//2),
            nn.Dropout2d(drop_prob),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=3//2),
            nn.Dropout2d(drop_prob),
            nn.Tanh(),
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
            nn.Dropout2d(drop_prob),
            nn.PixelShuffle(scale_factor)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x


class ESPCN_deconv(nn.Module):
    def __init__(self, scale_factor, num_channels=1, drop_prob=0.0, **kwargs):
        super().__init__()
        if scale_factor != 4:
            warnings.warn('WATCH OUT. This class is hardcoded for a scale factor of 4. Unexpected outputsizes will occur when not using this factor.')

        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=5 // 2),
            nn.Dropout2d(drop_prob),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=3 // 2),
            nn.Dropout2d(drop_prob),
            nn.Tanh(),
        )
        # This is going to be an ugly solution.
        # Hardcoded for an scaling factor of 4. Screw you
        self.last_part = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=1, padding=0),
            nn.Dropout2d(drop_prob),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(8, 1, kernel_size=3, padding=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x


class ESPCN_xconv(nn.Module):
    def __init__(self, scale_factor, num_channels=1, **kwargs):
        super().__init__()
        self.first_part = nn.Sequential(
            hlayers.xCNN(num_channels, 64, kernel_size=5, padding=5//2),
            nn.Tanh(),
            hlayers.xCNN(64, 32, kernel_size=3, padding=3//2),
            nn.Tanh(),
        )
        self.last_part = nn.Sequential(
            hlayers.xCNN(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(scale_factor)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x


class ESPCN_reversed(nn.Module):
    def __init__(self, scale_factor, num_channels=1, **kwargs):
        super().__init__()
        self.first_part = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=3 // 2),
            nn.Tanh(),
            nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2),
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(1, num_channels * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(1, 64, kernel_size=5, padding=5 // 2),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.last_part(x)
        x = self.first_part(x)
        return x


class ESPCN_normalisation(nn.Module):
    def __init__(self, scale_factor, num_channels=1, normalisation='batchnorm2d', drop_prob=0.2, **kwargs):
        super().__init__()
        self.first_part = nn.Sequential(
            mblocks.ConvBlock2D(in_chans=num_channels, out_chans=64, kernel_size=5, padding=5//2,
                                block_normalization=normalisation, block_activation='tanh', drop_prob=drop_prob),
            mblocks.ConvBlock2D(in_chans=64, out_chans=32, kernel_size=3, padding=3 // 2,
                                block_normalization=normalisation, block_activation='tanh', drop_prob=drop_prob),
        )
        self.last_part = nn.Sequential(
            mblocks.ConvBlock2D(in_chans=32, out_chans=num_channels * (scale_factor ** 2), kernel_size=3, padding=3 // 2,
                                block_normalization=normalisation, block_activation='tanh', drop_prob=drop_prob),
            nn.PixelShuffle(scale_factor)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x


if __name__ == "__main__":
    import torch
    import data_generator.Default as default_gen
    import helper.plot_fun as hplotf

    sel_scale = 4
    data_gen = default_gen.DataGeneratorSR(ddata='/home/bugger/Documents/data/celeba', input_shape=(256, 256), debug=True,
                                           file_ext='jpg', n_kernel=5, n_scale=sel_scale)

    container = data_gen.__getitem__(0)
    model_obj = ESPCN(sel_scale)

    with torch.no_grad():
        res = model_obj(container['input'][None, None])

    hplotf.plot_3d_list([res - container['target'][:, :255, :255], container['input'] ])

    # # Different version
    model_obj = ESPCN_reversed(sel_scale)

    derp = nn.PixelShuffle(3)
    derp(container['input'][None, None])
    with torch.no_grad():
        res = model_obj(container['input'][None, None])

    hplotf.plot_3d_list([res - container['target'][:, :255, :255], container['input']])

    # # With normalisation
    model_obj = ESPCN_normalisation(sel_scale)

    with torch.no_grad():
        res = model_obj(container['input'][None, None])

    hplotf.plot_3d_list([res - container['target'][:, :255, :255], container['input']])

    # # With xcnn
    model_obj = ESPCN_xconv(sel_scale)

    with torch.no_grad():
        res = model_obj(container['input'][None])

    hplotf.plot_3d_list([res, container['input']])

    # # With deconv
    model_obj = ESPCN_deconv(sel_scale)

    with torch.no_grad():
        res = model_obj(container['input'][None])

    hplotf.plot_3d_list([res, container['input']])
