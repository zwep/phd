from tooling import tsne
import pydicom
from model.VGG import Vgg16
from model.DenseNet import DenseNetFeatures
import torch
import helper.array_transf as harray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import skimage.transform as sktransform


class TSNEPlot:
    """
    Class that facilitates the plotting of a bunch of data with TSNE
    """

    def __init__(self, data_array, data_labels, model_name='vgg', feature_layer=2):
        # Assuming data array is of shape N, nx, ny
        # Assuming label array is of shape N, 1
        self.data_array = data_array
        self.data_labels = data_labels
        self.unique_labels = list(set(data_labels))
        self.n_labels = len(set(data_labels))
        self.n_images = len(data_array)

        self.model_obj = self.get_feature_model(model_name)
        self.feature_layer = feature_layer

        self.feature_array = np.array([self.get_feature_from_array(x) for x in data_array])
        print("Feature array has shape ", self.feature_array.shape)
        self.feature_tsne = tsne.tsne(self.feature_array)

    def get_feature_model(self, model_name):
        if model_name == 'vgg':
            model_obj = Vgg16(requires_grad=False).float()
        elif model_name == 'dense':
            model_obj = DenseNetFeatures(requires_grad=False).float()
        else:
            model_obj = None
            print("Unknown model name ", model_name)
            print("Choose : vgg or dense")
        return model_obj

    def get_feature_from_array(self, x):
        np_array = harray.scale_minmax(x)
        np_array = sktransform.resize(np_array, (256, 256), preserve_range=True, anti_aliasing=False)
        tens_array = torch.from_numpy(np_array)[None, None].float()
        tens_features = self.model_obj(tens_array)
        avg_features = np.array(torch.mean(tens_features[self.feature_layer], dim=1)).ravel()
        return avg_features

    def plot_features_tsne(self, feature_array=None):
        if feature_array is None:
            feature_array = self.feature_tsne

        cmap = matplotlib.cm.get_cmap('plasma', lut=self.n_labels)
        fig, ax = plt.subplots()
        old_label = 'Nothing'
        for i, i_label in enumerate(self.data_labels):
            sel_color = self.unique_labels.index(i_label)
            if old_label != i_label:
                plt.scatter(feature_array[i, 0], feature_array[i, 1], color=cmap(sel_color), label=i_label)
                old_label = i_label
            else:
                plt.scatter(feature_array[i, 0], feature_array[i, 1], color=cmap(sel_color))

        plt.legend()
        return fig


if __name__ == "__main__":
    # Get some data...
    A = np.random.normal(1, size=(10, 128, 128))
    label_A = ['gaussian'] * len(A)
    B = np.random.rayleigh(1, size=(10, 128, 128))
    label_B = ['rayleigh'] * len(B)
    C = np.concatenate([A, B])
    label_C = label_A + label_B

    import helper.plot_class as hplotc
    hplotc.SlidingPlot(C)
    tsne_obj = TSNEPlot(C, data_labels=label_C)
    tsne_obj.plot_features_tsne()