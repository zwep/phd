import os
import helper.array_transf as harray
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import data
import matplotlib.pyplot as plt
import harreveltools

"""

Here we develop GLCM helpers..

"""


class StabilityGLCM:
    def __init__(self, image) -> None:
        self.PATCH_SIZE = 21
        self.DISTANCE = [5]
        self.ANGLE = [0]
        self.LEVELS = 256
        self.N_ITER = int(1e3)
        self.image = image

        # define grass x-y locations (and below that sky location)
        # Used in a specific image...
        self.grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
        self.sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]

    def get_patch(self, x, y):
        return image[x : x + self.PATCH_SIZE, y : y + self.PATCH_SIZE]

    def get_patches(self, coord_list):
        # Can be useful for debugging purpose
        return [self.get_patch(*x) for x in coord_list]

    def get_GLCM_obj(self, patch, levels=None):
        if levels is None:
            levels = self.LEVELS
        return graycomatrix(
            patch,
            distances=self.DISTANCE,
            angles=self.ANGLE,
            levels=levels,
            symmetric=True,
            normed=True,
        )

    def _get_GLCM_metric(self, glcm_obj, metric):
        # "dissimilarity"
        # "correlation"
        return graycoprops(glcm_obj, metric)

    def get_GLCM_metric(self, patch, metric):
        # "dissimilarity"
        # "correlation"
        glcm_obj = self.get_GLCM_obj(patch)
        return graycoprops(glcm_obj, metric)

    def stability_analysis_random_noise(self, patch, metric):
        # Comparisson
        main_glcm_metric = self.get_GLCM_metric(patch, metric)[0][0]
        metric_dstr = []
        for _ in range(self.N_ITER):
            alt_patch = patch + np.random.randint(
                0, 2, size=patch.shape, dtype=np.uint8
            )
            alt_patch[alt_patch < 0] = 0
            alt_patch[alt_patch > 255] = 255

            glcm_metric = self.get_GLCM_metric(alt_patch, metric)[0][0]
            metric_dstr.append(glcm_metric)
        stab_error = main_glcm_metric - np.mean(metric_dstr)
        rel_stab_error = stab_error / main_glcm_metric
        return stab_error, rel_stab_error

    def stability_analysis_worst_noise(self, patch, metric):
        # Comparisson
        main_glcm_metric = self.get_GLCM_metric(patch, metric)[0][0]
        #
        residual = patch % 1
        patch[residual >= 0.5] = np.floor(patch[residual >= 0.5])
        patch[residual < 0.5] = np.floor(patch[residual < 0.5])
        alt_glcm_metric = self.get_GLCM_metric(patch, metric)[0][0]
        stab_error = main_glcm_metric - alt_glcm_metric
        rel_stab_error = stab_error / main_glcm_metric
        return stab_error, rel_stab_error


if __name__ == "__main__":
    # open the camera image
    image = data.camera()

    stab_obj = StabilityGLCM(image=image)

    # Get patches
    patch_shape = tuple(np.array(image.shape) // 10)
    stride = min(patch_shape)
    temp_patches = harray.get_patches(image, patch_shape=patch_shape, stride=stride)

    stab_error_dstr = []
    rel_stab_error_dstr = []
    for sel_index, sel_patch in enumerate(temp_patches):
        print(f'{sel_index} / {len(temp_patches)}', end='\r')
        stab_error, rel_stab_error = stab_obj.stability_analysis_random_noise(sel_patch, metric="homogeneity")
        stab_error_dstr.append(stab_error)
        rel_stab_error_dstr.append(rel_stab_error)

    print(stab_error_dstr)
    print(rel_stab_error_dstr)
    plt.hist(rel_stab_error_dstr)
    plt.hist(stab_error_dstr)

    worst_stab_error_dstr = []
    worst_rel_stab_error_dstr = []
    for sel_index, sel_patch in enumerate(temp_patches):
        print(f'{sel_index} / {len(temp_patches)}', end='\r')
        stab_error, rel_stab_error = stab_obj.stability_analysis_worst_noise(sel_patch, metric="energy")
        worst_stab_error_dstr.append(stab_error)
        worst_rel_stab_error_dstr.append(rel_stab_error)

    plt.hist(worst_rel_stab_error_dstr)

    main_glcm_metric = stab_obj.get_GLCM_metric(patch, metric)[0][0]
    #
    patch = np.copy(sel_patch)
    residual = patch % 1
    patch[residual >= 0.5] = np.floor(patch[residual >= 0.5])
    patch[residual < 0.5] = np.floor(patch[residual < 0.5])
    import helper.plot_class as hplotc
    hplotc.ListPlot([patch, sel_patch])
    alt_glcm_metric = self.get_GLCM_metric(patch, metric)[0][0]
    stab_error = main_glcm_metric - alt_glcm_metric
    rel_stab_error = stab_error / main_glcm_metric