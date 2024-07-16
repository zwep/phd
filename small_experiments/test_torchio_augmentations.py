import torchio
import data_generator.Default as data_gen_default
"""
Test the torch io augmentations
"""

data_gen = data_gen_default.DataGeneratorNoise(ddata='/media/bugger/MyBook/data/celeba', input_shape=(64, 64),
                                               debug=True, file_ext='jpg', n_rep=100, noise_mode='additive')

container = data_gen.__getitem__(1)

dir(torchio.transforms.spatial_transform)
dir(torchio.transforms)
dir(torchio)

# Easy transforms
torchio_transform = torchio.Compose([torchio.RandomFlip(), torchio.RandomBlur(), torchio.RandomNoise()])

# More intense transforms
torchio_transform = torchio.Compose([torchio.RandomBiasField(), torchio.RandomElasticDeformation()])

torchio_transform(container['input'][None]).shape
