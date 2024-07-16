from model.VGG import Vgg16
import sklearn.decomposition
from model.DenseNet import DenseNetFeatures
import torch
import numpy as np

"""

"""


class GetEmbedding:
    def __init__(self, data_array, model_name='vgg', feature_layer=2):
        # Assuming data array is of shape N, nx, ny
        self.data_array = data_array
        self.n_images = len(data_array)
        self.model_obj = self.get_feature_model(model_name)
        self.feature_layer = feature_layer
        self.model_name = model_name
        self.feature_list = [self.get_feature_from_array(x) for x in data_array]
        if model_name == 'vgg':
            self.n_components = (feature_layer + 1) * 64
        else:
            self.n_components = 64
        self.PCA_obj = sklearn.decomposition.PCA(n_components=self.n_components)

    def get_feature_model(self, model_name):
        if model_name == 'vgg':
            model_obj = Vgg16(requires_grad=False).float()
        elif model_name == 'dense':
            model_obj = DenseNetFeatures(requires_grad=False).float()
        elif model_name == 'resnet':
            model_obj = ...
        elif model_name == 'nnunet_gan':
            model_obj = ...
        elif model_name == 'nnunet_biasf':
            model_obj = ...
        else:
            model_obj = None
            print("Unknown model name ", model_name)
            print("Choose : vgg or dense")
        return model_obj

    def get_feature_from_array(self, x):
        tens_array = torch.from_numpy(x)[None, None].float()
        tens_features = self.model_obj(tens_array)
        all_features = tens_features[self.feature_layer]
        return all_features

    def get_pca_feature_array(self, n_pca=None):
        pca_feature_array = []
        for x_feature_map in self.feature_list:
            x_feature_map = np.squeeze(x_feature_map.numpy())
            orig_shape = x_feature_map.shape[-2:]
            n_channels = x_feature_map.shape[0]
            x_reshp = x_feature_map.reshape(n_channels, -1)
            self.PCA_obj.fit(x_reshp)
            cumulative_sum = np.cumsum(self.PCA_obj.explained_variance_)
            cumulative_sum = cumulative_sum / cumulative_sum[-1] * 100
            max_n_components = np.argwhere(cumulative_sum > 95).ravel()[0]
            selected_components = self.PCA_obj.components_[:max_n_components]
            x_pca_obj = selected_components.reshape((max_n_components, ) + orig_shape)
            pca_feature_array.append(x_pca_obj)
        return pca_feature_array


if __name__ == "__main__":
    from skimage.data import astronaut, camera
    import skimage.transform
    import helper.plot_class as hplotc
    import matplotlib.pyplot as plt
    A = astronaut()[:, :, 0]
    B = camera()
    img_list = [A, B]
    img_list = [skimage.transform.resize(x, (256, 256)) for x in img_list]
    vgg_obj = GetEmbedding(data_array=img_list, feature_layer=2)
    pca_features = vgg_obj.get_pca_feature_array()
    min_pca_comp = min([x.shape[0] for x in pca_features])
    truncated_pca_features = [x[:min_pca_comp] for x in pca_features]
