import sys
import sklearn.neighbors
import os
from IPython.display import display, clear_output, HTML
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


username = os.environ.get('USER', os.environ.get('USERNAME'))
remote = False
if username != 'bugger':
    remote = True

ddest = '/home/bugger/Documents/TUE/AssistentDocent/Assignment2'
if remote:
    ddest = '/data/seb/course_8dc00'

mia_path_code = '/home/bugger/PycharmProjects/8dc00-mia-private/code'
mia_path_data = '/home/bugger/PycharmProjects/8dc00-mia-private/data'
if remote:
    mia_path_code = '/data/seb/code/8dc00-mia-private/code'
    mia_path_data = '/data/seb/code/8dc00-mia-private/data'

data_path = os.path.join(mia_path_data, 'nuclei_data_classification.mat')

sys.path.append(mia_path_data)
sys.path.append(mia_path_code)
import cad_util as util

"""
We also need to prepare task 6..

1. Reduce the dimensions of the training set using PCA. Next, use k-NN to classify a new image from the test dataset.

"""


def get_transformed_images(mat_obj, sel_key):
    x_images = mat_obj[f"{sel_key}_images"]
    n_images = x_images.shape[-1]
    x_images_reshape = x_images.T.reshape(n_images, -1)
    x_y = mat_obj[f"{sel_key}_y"]
    x_mean = np.mean(x_images_reshape, axis=0).reshape(1, -1)
    x_std = np.std(x_images_reshape, axis=0).reshape(1, -1)
    x_images_reshape = (x_images_reshape - x_mean) / x_std
    return x_images, x_images_reshape, x_y


def train_test_knn(train_images, train_target, test_images, test_target, k=5):
    knn_obj = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    knn_obj.fit(train_images, train_target)
    test_prediction = knn_obj.predict(test_images)
    confusion_matrix = sklearn.metrics.confusion_matrix(test_target, test_prediction)
    return knn_obj, confusion_matrix


def get_metric_confusion_matrix(confusion_matrix):
    total_cases = np.sum(confusion_matrix)
    tp_tf = np.diag(confusion_matrix)
    accuracy = np.sum(tp_tf) / total_cases
    print('Accuracy ', accuracy)
    denom_specificity = confusion_matrix_normal[:, 1]
    nom_specificity = confusion_matrix_normal[1, 1]
    specificity = nom_specificity / denom_specificity
    print('Specificity ', nom_specificity / denom_specificity)
    #
    denom_sensitivity = confusion_matrix_normal[:, 0]
    nom_sensitivity = confusion_matrix_normal[0, 0]
    sensitivity = nom_sensitivity / denom_sensitivity
    print('Sensitivity ', nom_sensitivity / denom_sensitivity)
    return {'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity}


def get_optimal_pca(x, max_samples, cut_off_perc=95):
    n_train, n_features = x.shape
    if max_samples is None:
        max_samples = n_train
    sel_comp = min(n_features, max_samples)
    pca = PCA(n_components=sel_comp)
    pca.fit(x[:max_samples])
    cumulative_sum = np.cumsum(pca.explained_variance_)
    cumulative_sum = cumulative_sum / cumulative_sum[-1] * 100
    n_optimal_components = np.where(cumulative_sum > cut_off_perc)[0][0]
    return n_optimal_components


def get_pca_obj(x, n_comp):
    pca_obj = PCA(n_components=n_comp)
    pca_obj.fit(x)
    return pca_obj



# To reduce the training time..
n_max_train = None#2000
n_max_test = None#300

if remote:
    n_max_train = None
    n_max_test = None

# Load the data
mat_obj = scipy.io.loadmat(data_path)

training_images, training_images_reshape, training_y = get_transformed_images(mat_obj, 'training')
# Validation is not really used...
# validation_images, validation_images_reshape, validation_y = get_transformed_images(mat_obj, 'validation')
test_images, test_images_reshape, test_y = get_transformed_images(mat_obj, 'test')

# Reduce dimension using PCA for train and test
n_optimal = get_optimal_pca(training_images_reshape, n_max_train)
pca_obj_normal = get_pca_obj(training_images_reshape[:n_max_train], n_optimal)
training_images_pca = pca_obj_normal.transform(training_images_reshape)
test_images_pca = pca_obj_normal.transform(test_images_reshape)

range_neighbors = range(5, 200, 10)
accuracy_list = []
accuracy_list_pca = []

for k in range_neighbors:
    knn_normal, confusion_matrix_normal = train_test_knn(training_images_reshape[:n_max_train],
                                                         training_y[:n_max_train, 0],
                                                         test_images_reshape[:n_max_test],
                                                         test_y[:n_max_test, 0].astype(float),
                                                         k=k)
    print('Confusing matrix using normal data')
    print(confusion_matrix_normal)
    metric_dict = get_metric_confusion_matrix(confusion_matrix_normal)
    accuracy_list.append(metric_dict['accuracy'])


    knn_pca, confusion_matrix_pca = train_test_knn(training_images_pca[:n_max_train],
                                                   training_y[:n_max_train, 0],
                                                   test_images_pca[:n_max_test],
                                                   test_y[:n_max_test, 0].astype(float),
                                                   k=k)

    print('Confusing matrix using PCA data')
    print(confusion_matrix_pca)
    metric_dict = get_metric_confusion_matrix(confusion_matrix_pca)
    accuracy_list_pca.append(metric_dict['accuracy'])

fig, ax = plt.subplots()
ax.plot(range_neighbors, accuracy_list, 'r', label='accuracy normal')
ax.plot(range_neighbors, accuracy_list_pca, 'b', label='accuracy normal')
plt.legend()
fig.savefig(os.path.join(ddest, 'accuracy.png'))