import torch
import helper.plot_class as hplotc
import skimage.transform as sktransf
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torchvision import transforms
import numpy as np
import skimage.data
import matplotlib.pyplot as plt


class InceptionV3(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.inception_model = models.inception_v3(pretrained=pretrained)
        self.inception_model.eval()
        self.inception_model.fc = nn.Identity()

        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, bias=False)
        conv_weights = torch.from_numpy(np.ones((3, 1, 1, 1))).float()
        self.conv_layer.weight = torch.nn.Parameter(conv_weights, requires_grad=False)

    def preproc(self, X):
        preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(X)
        # input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        return input_tensor

    def forward(self, X):
        with torch.no_grad():
            X = self.conv_layer(X)
            X = self.preproc(X)
            res = self.inception_model(X)
        return res


if __name__ == "__main__":
    A = skimage.data.astronaut()[:, :, 0]
    A = sktransf.resize(A, output_shape=(299, 299), preserve_range=True, anti_aliasing=False)
    A_tens = torch.as_tensor(A).float()[None, None]

    derp = InceptionV3()
    res = derp(A_tens)
    plt.hist(res)