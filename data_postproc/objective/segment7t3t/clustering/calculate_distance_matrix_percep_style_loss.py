from data_postproc.objective.segment7t3t.clustering.FileGatherer import FileGather
from model.VGG import Vgg16
import numpy as np

# New method..
import torch
import torch.nn as nn
import helper_torch.loss as hloss

"""
Here we calculate the pereceptual Style loss
"""


class PerceptualLossStyleSeparateLoss(nn.Module):
    """
    Computes stuff
    """
    def __init__(self, vgg_model):
        super().__init__()
        self.vgg = vgg_model

    @staticmethod
    def gram_matrix(x_tens):
        """ Calculate the Gram Matrix of a given tensor
            Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        """
        # get the batch_size, depth, height, and width of the Tensor
        b, d, h, w = x_tens.size()
        # reshape so we're multiplying the features for each channel
        x_tens = x_tens.view(b, d, h * w)
        # calculate the gram matrix
        gram = torch.einsum("bcr, brd->bcd", x_tens, x_tens.transpose(-1, 1)) / (d * h * w)
        return gram
    def forward(self, y_pred, y_true):
        loss_obj_mse = torch.nn.MSELoss()
        frobenius_loss = hloss.FrobeniusLoss()
        loss_style = 0
        loss_perc = 0
        pred_model_split = torch.split(y_pred, 1, dim=1)
        true_model_split = torch.split(y_true, 1, dim=1)
        iter = 0
        for i_chan in range(len(pred_model_split)):
            features_pred = self.vgg(pred_model_split[i_chan])
            features_target = self.vgg(true_model_split[i_chan])
            for x_ftr, y_ftr in zip(features_pred, features_target):
                iter += 1
                loss_value_ftr = loss_obj_mse(x_ftr, y_ftr)
                gram_x_ftr = self.gram_matrix(x_ftr)
                gram_y_ftr = self.gram_matrix(y_ftr)
                style_loss = frobenius_loss(gram_x_ftr, gram_y_ftr)
                loss_style += style_loss
                loss_perc += loss_value_ftr
        loss_perc /= iter
        loss_style /= iter
        return loss_perc, loss_style


def get_metric(X, loss_obj, device=None):
    # I Would like to get these metrics (SSIM, WSS, Contrast) on the feature maps as well
    if device is None:
        device = 'cpu'
    n_img = X.shape[0]
    style_loss = np.zeros((n_img, n_img))
    perception_loss = np.zeros((n_img, n_img))
    for i_img in range(n_img):
        print(f"{i_img} / {n_img}", end='\r')
        x_i = X[i_img]
        x_i_tens = torch.from_numpy(x_i).float()[None, None].to(device)
        for j_img in range(i_img, n_img):
            x_j = X[j_img]
            x_j_tens = torch.from_numpy(x_j).float()[None, None].to(device)
            loss_perc, loss_style = loss_obj(x_i_tens, x_j_tens)
            style_loss[i_img, j_img] = loss_style
            perception_loss[i_img, j_img] = loss_perc
    return perception_loss, style_loss


if __name__ == "__main__":
    file_gather_obj = FileGather()
    # data_array = file_gather_obj.array_list
    # prefix = ''
    data_array = file_gather_obj.cropped_array_list
    prefix = 'cropped_'
    with open('/data/seb/data/similarity_matrix/key_length_string.txt', 'w') as f:
        f.write(file_gather_obj.string_array_key)

    index_gpu = 0
    device = torch.device("cuda:{}".format(str(index_gpu)) if torch.cuda.is_available() else "cpu")
    vgg_obj = Vgg16().to(device)
    loss_obj = PerceptualLossStyleSeparateLoss(vgg_model=vgg_obj)
    perception_loss, style_loss = get_metric(data_array, loss_obj=loss_obj, device=device)
    np.save(f'/data/seb/data/similarity_matrix/{prefix}distance_perception.npy', perception_loss)
    np.save(f'/data/seb/data/similarity_matrix/{prefix}distance_style.npy', style_loss)