import tensorflow as tf

from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet

# from objective_configuration.reconstruction import DWEIGHTS_NCPD

DWEIGHTS_NCPD = '/local_scratch/sharreve/mri_data/pretrained_networks/ncpdnet/model_weights.h5'

model = NCPDNet(
    three_d=True,
    n_iter=6,
    n_filters=16,
    im_size=(176, 256, 256),
    dcomp=True,
    fastmri=False,
)

kspace_shape = 1
inputs = [
    tf.zeros([1, 1, kspace_shape, 1], dtype=tf.complex64),
    tf.zeros([1, 3, kspace_shape], dtype=tf.float32),
    (tf.constant([(176, 256, 256)]), tf.ones([1, kspace_shape], dtype=tf.float32)),
]

# Source: https://huggingface.co/zaccharieramzi/NCPDNet-3D
# pip install git+https://github.com/zaccharieramzi/fastmri-reproducible-benchmark

z = model(inputs)
model.load_weights(DWEIGHTS_NCPD)


