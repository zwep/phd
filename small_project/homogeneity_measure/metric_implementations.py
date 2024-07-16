import helper.array_transf as harray
import numpy as np
import scipy.io
import skimage.data
import skimage.transform
import skimage.metrics
import scipy.integrate
from skimage.util import img_as_ubyte
import skimage.feature
import scipy.optimize


class MinimizeL2:
    def __init__(self, x, y):
        # Scale X to Y
        self.x = x
        self.y = y

    def calc_solution(self, coefficients):
        a, b = coefficients
        return np.linalg.norm(a * self.x + b - self.y)

    def minimize_run(self):
        opt_obj = scipy.optimize.minimize(fun=self.calc_solution, x0=np.array([1, 0]), tol=1e-8)
        a, b = opt_obj.x
        return a, b

    def get_transform(self, a, b):
        return self.x * a + b


class MinimizeL2Line:
    # Finds a linear combination to align x with y
    # Input can be 2D images, uses a centroid line
    def __init__(self, x, y):
        # Scale X to Y
        self.x = x
        self.y = y
        nx, ny = x.shape
        self.x_line = self.x[nx//2]
        self.y_line = self.y[nx//2]

    def calc_solution(self, coefficients):
        a, b = coefficients
        return np.linalg.norm(a*self.x_line + b - self.y_line)

    def minimize_run(self):
        opt_obj = scipy.optimize.minimize(fun=self.calc_solution, x0=np.array([1, 0]), tol=1e-8)
        a, b = opt_obj.x
        return a, b

    def get_transform(self, a, b):
        return self.x * a + b

    def get_optimal_transform(self):
        result_min = self.minimize_run()
        x_transformed = self.get_transform(*result_min)
        return x_transformed


class MinimizeL2Points:
    # Combines multiple lines and makes them `closer` to each other
    def __init__(self, x):
        # Should be of shape... (n_models, n_points)
        self.x = x
        self.init_norm = np.linalg.norm(self.get_dist_matrix(self.x)) // 2
        self.n_models = x.shape[0]

    def get_dist_matrix(self, x):
        n_points, n_dim = x.shape
        dist_matrix = np.zeros((n_points, n_points))
        for ii in range(n_dim):
            dist_matrix += ((x[:, ii:ii + 1] - x[:, ii:ii + 1].T)) ** 2
        dist_matrix = np.sqrt(dist_matrix)
        return dist_matrix

    def calc_solution(self, coefficients):
        a, b = np.split(coefficients, 2)
        x_trans = (self.x.T @ np.diag(a) + b).T
        return np.linalg.norm(self.get_dist_matrix(x_trans)) + self.init_norm * np.linalg.norm(1-a) + np.linalg.norm(b)

    def minimize_run(self):
        init = np.concatenate([np.ones(self.n_models), np.zeros(self.n_models)])
        opt_obj = scipy.optimize.minimize(fun=self.calc_solution, x0=init, tol=1e-8)
        # a, b = opt_obj.x
        # return a, b
        return opt_obj

    def get_transform(self, coefficients):
        a, b = np.split(coefficients, 2)
        return (self.x.T @ np.diag(a) + b).T


def get_hi_value_integral(x, mask, selected_powers=None):
    # x is the biasfield corrected image
    # y is the starting/reference image
    if selected_powers is None:
        selected_powers = np.arange(0, 20, 0.1)

    power_mean_value_x = [(x[mask == 1] ** n_power).mean() for n_power in selected_powers]
    abs_hi_norm = scipy.integrate.simps(power_mean_value_x, selected_powers)
    return abs_hi_norm


def get_glcm_patch_object(x, patch_size=32, glcm_dist=1, angles=None, n_angles=8):
    # Use patches and not the full image to be better able to compare tissue types
    # Patches should be not overlapping
    if isinstance(glcm_dist, int):
        glcm_dist = [glcm_dist]
    if angles is None:
        # Need to leave out the last one.. Otherwise counting both 0 and 360 degrees
        # Couldve also done an np range with the proper increment... ah well
        angles = np.linspace(0, 2 * np.pi - (2 * np.pi / n_angles), n_angles)

    stride = patch_size
    x_patches = harray.get_patches(x, patch_shape=(patch_size, patch_size), stride=stride)
    # Scale them to 0..1 to limit any global intensity differences...
    x_patches_bytes = [img_as_ubyte(harray.scale_minmax(x)) for x in x_patches]
    glcm = [skimage.feature.graycomatrix(x, distances=glcm_dist, angles=angles, levels=256, symmetric=True, normed=True) for x in x_patches_bytes]
    return glcm


def get_glcm_features(glcm_obj, feature_keys=None, key_appendix=None):
    if key_appendix is None:
        key_appendix = ''

    if feature_keys is None:
        feature_keys = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    if isinstance(feature_keys, str):
        feature_keys = [feature_keys]

    feature_dict = {}
    for i_feature in feature_keys:
        feature_level = np.mean([skimage.feature.graycoprops(x, i_feature) for x in glcm_obj])
        feature_dict[i_feature + key_appendix] = feature_level

    return feature_dict


def get_relative_glcm_features(input, target, patch_size=None, glcm_dist=None, feature_keys=None):
    # Input = inhomogeneous image
    # Target = homogeneous image
    if patch_size is None:
        patch_size = min(input.shape) // 3
    elif patch_size == 'max':
        patch_size = min(input.shape)
    if glcm_dist is None:
        glcm_dist = [1, 2]
    if feature_keys is None:
        feature_keys = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    glcm_obj_input = get_glcm_patch_object(input, patch_size=patch_size, glcm_dist=glcm_dist)
    glcm_obj_target = get_glcm_patch_object(target, patch_size=patch_size, glcm_dist=glcm_dist)
    feature_input_dict = {}
    feature_target_dict = {}
    feature_rel_dict = {}
    n_patches = float(len(glcm_obj_target))
    for input_patch_obj, target_patch_obj in zip(glcm_obj_input, glcm_obj_target):
        for i_feature in feature_keys:
            feature_rel_dict.setdefault(i_feature, 0)
            feature_target_dict.setdefault(i_feature, 0)
            feature_input_dict.setdefault(i_feature, 0)
            target_feature_value = skimage.feature.graycoprops(target_patch_obj, i_feature)
            input_feature_value = skimage.feature.graycoprops(input_patch_obj, i_feature)
            rel_change = (target_feature_value - input_feature_value) / input_feature_value
            feature_rel_dict[i_feature] += np.mean(rel_change) / n_patches
            feature_target_dict[i_feature] += np.mean(target_feature_value) / n_patches
            feature_input_dict[i_feature] += np.mean(input_feature_value) / n_patches
    return feature_rel_dict, feature_input_dict, feature_target_dict


def local_homog_lukasiewicz_implication(x, order=1.):
    # Officially it is defined as I(x, y) = min(1-x+y, 1)
    # Used orders are 1, 2, and 1/2 (in the paper)
    local_homog = (1 - np.min(x) + np.max(x)) ** order
    return local_homog


def local_homog_mean_aggregator(x):
    local_homog = (1 - np.sqrt(np.mean((x - np.mean(x)) ** 2)))
    return local_homog


def local_homog_median_aggregator(x):
    l = len(x.ravel())
    if l % 2 == 0:
        k = l // 2
        # Normally first index is lowest..
        k_th_largest_index = np.argsort(x.ravel())[::-1][k - 1]
        idx, idy = np.unravel_index(k_th_largest_index, x.shape)
        k_th_largest_value = x[idx, idy]
        k_th_p1_largest_index = np.argsort(x.ravel())[::-1][k]
        idx, idy = np.unravel_index(k_th_p1_largest_index, x.shape)
        k_th_p1_largest_value = x[idx, idy]
        m_median = (k_th_largest_value + k_th_p1_largest_value) / 2
    else:
        k = (l - 1) // 2
        k_th_largest_index = np.argsort(x.ravel())[::-1][k - 1]
        idx, idy = np.unravel_index(k_th_largest_index, x.shape)
        k_th_largest_value = x[idx, idy]
        m_median = k_th_largest_value

    local_homog = (1 - np.sqrt(np.mean(x - m_median) ** 2))
    return local_homog


def get_fuzzy_features(x, patch_size=32, stride=None, key_appendix=None):
    if key_appendix is None:
        key_appendix = ''

    if stride is None:
        stride = patch_size // 2

    temp_patches = harray.get_patches(x, patch_shape=(patch_size, patch_size), stride=stride)
    temp_patches = np.array([x for x in temp_patches if (np.isclose(x, 0, atol=1e-5)).sum() / np.prod(x.shape) < 0.5])
    # mean_agg = np.mean([local_homog_mean_aggregator(x) for x in temp_patches])
    luka_order_half = np.mean([local_homog_lukasiewicz_implication(x, order=0.5) for x in temp_patches])
    luka_order_one = np.mean([local_homog_lukasiewicz_implication(x, order=1) for x in temp_patches])
    luka_order_two = np.mean([local_homog_lukasiewicz_implication(x, order=2) for x in temp_patches])
    # median_agg = np.mean([local_homog_median_aggregator(x) for x in temp_patches])

    # Removed the mean and median agg
    # temp_fuzzy = {'mean_agg' + key_appendix: mean_agg,
    #               'median_agg' + key_appendix: median_agg,
    #               'luka_half' + key_appendix: luka_order_half,
    #               'luka_one' + key_appendix: luka_order_one,
    #               'luka_two' + key_appendix: luka_order_two}
    temp_fuzzy = {'luka_half' + key_appendix: luka_order_half,
                  'luka_one' + key_appendix: luka_order_one,
                  'luka_two' + key_appendix: luka_order_two}
    return temp_fuzzy


def get_fuzzy_luka_order(x, patch_size=32, order=2, stride=None):
    if stride is None:
        stride = patch_size // 2

    temp_patches = harray.get_patches(x, patch_shape=(patch_size, patch_size), stride=stride)
    temp_patches = np.array([x for x in temp_patches if (np.isclose(x, 0, atol=1e-5)).sum() / np.prod(x.shape) < 0.5])
    luka_order_two = np.mean([local_homog_lukasiewicz_implication(x, order=order) for x in temp_patches])

    return luka_order_two

