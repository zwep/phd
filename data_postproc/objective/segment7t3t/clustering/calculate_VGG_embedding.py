from data_postproc.objective.segment7t3t.clustering.FileGatherer import FileGather
from helper.clustering import GetEmbedding
from model.VGG import Vgg16
import sklearn.decomposition
from model.DenseNet import DenseNetFeatures
import torch
import numpy as np

"""
Since I am not sure about the metrics I used to calculate a difference...
Lets see what this has to offer...

"""


file_gather_obj = FileGather()
# data_array = file_gather_obj.array_list
# prefix = ''
data_array = file_gather_obj.cropped_array_list
prefix = 'cropped_'
with open('/data/seb/data/similarity_matrix/key_length_string.txt', 'w') as f:
    f.write(file_gather_obj.string_array_key)

i_layer = 2
for i_layer in range(4):
    vgg_obj = GetEmbedding(data_array, feature_layer=i_layer)
    pca_feature_array = np.array(vgg_obj.get_pca_feature_array())
    string_name = f'{vgg_obj.model_name}_{vgg_obj.feature_layer}'
    np.save(f'/data/seb/data/similarity_matrix/{prefix}pca_feature_array_{string_name}.npy', pca_feature_array)

