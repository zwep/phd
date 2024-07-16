import tensorflow as tf
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow.keras.layers as tfl
import tensorflow.contrib.layers as tf_cl
from PIL import Image
import concurrent.futures
import glob
from cv2 import imwrite
import cv2
import random

def l1loss(x, y):

    return tf.reduce_mean(tf.abs(x - y))

def l2loss(x, y):
    return tf.reduce_mean(tf.squared_difference(x, y))


def DSSIMloss(real, gen, max_val=1):
    ssim = tf.image.ssim(real, gen, max_val=1)
    dssim = tf.reduce_mean(((1.0 - ssim) / 2))

    return dssim


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x)


def instance_norm(x, epsilon=1e-5):
    with tf.variable_scope('instance_norm',reuse=tf.AUTO_REUSE):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable(
            name='scale',
            shape=[x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable(
            name='offset',
            shape=[x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset
        return out

def conv2d(inputs,
           activation_fn=lrelu,
           normalizer_fn=instance_norm,
           scope='conv2d',
           **kwargs):
    with tf.variable_scope(scope or 'conv2d'):
        h = tf_cl.conv2d(
            inputs=inputs,
            activation_fn=None,
            normalizer_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            biases_initializer=None,
            **kwargs)
        if normalizer_fn:
            h = normalizer_fn(h)
        if activation_fn:
            h = activation_fn(h)
        return h

def generator(image, scope=None):
    with tf.variable_scope(scope or 'generator', reuse=tf.AUTO_REUSE):
        def dil_conv(l, in_channel):
            l_3X3 = tfl.Conv2D(in_channel, 3, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                               activation=tf.nn.relu)(l)
            l_3X3r2 = tfl.Conv2D(in_channel, 5, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                 activation=tf.nn.relu)(l)
            l_3X3r4 = tfl.Conv2D(in_channel, 9, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                 activation=tf.nn.relu)(l)
            l_3X3r6 = tfl.Conv2D(in_channel, 13, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                 activation=tf.nn.relu)(l)

            concat = tf.concat([l_3X3, l_3X3r2, l_3X3r4, l_3X3r6], 3)

            l_3X1 = tfl.Conv2D(in_channel, (3, 1), strides=(1, 1), padding='same', data_format=None,
                               dilation_rate=(1, 1), activation=tf.nn.relu)(l)
            l_1X3 = tfl.Conv2D(in_channel, (1, 3), strides=(1, 1), padding='same', data_format=None,
                               dilation_rate=(1, 1), activation=tf.nn.relu)(l)
            l_hv3 = tf.math.add(l_3X1, l_1X3)
            l_hv3 = tf.math.divide(l_hv3, 2)

            l_5X1 = tfl.Conv2D(in_channel, (5, 1), strides=(1, 1), padding='same', data_format=None,
                               dilation_rate=(1, 1), activation=tf.nn.relu)(l)
            l_1X5 = tfl.Conv2D(in_channel, (1, 5), strides=(1, 1), padding='same', data_format=None,
                               dilation_rate=(1, 1), activation=tf.nn.relu)(l)
            l_hv5 = tf.math.add(l_5X1, l_1X5)
            l_hv5 = tf.math.divide(l_hv5, 2)

            l_9X1 = tfl.Conv2D(in_channel, (9, 1), strides=(1, 1), padding='same', data_format=None,
                               dilation_rate=(1, 1), activation=tf.nn.relu)(l)
            l_1X9 = tfl.Conv2D(in_channel, (1, 9), strides=(1, 1), padding='same', data_format=None,
                               dilation_rate=(1, 1), activation=tf.nn.relu)(l)
            l_hv9 = tf.math.add(l_9X1, l_1X9)
            l_hv9 = tf.math.divide(l_hv9, 2)

            l_13X1 = tfl.Conv2D(in_channel, (13, 1), strides=(1, 1), padding='same', data_format=None,
                                dilation_rate=(1, 1), activation=tf.nn.relu)(l)
            l_1X13 = tfl.Conv2D(in_channel, (1, 13), strides=(1, 1), padding='same', data_format=None,
                                dilation_rate=(1, 1), activation=tf.nn.relu)(l)
            l_hv13 = tf.math.add(l_13X1, l_1X13)
            l_hv13 = tf.math.divide(l_hv13, 2)

            concat1 = tf.concat([l_hv3, l_hv5, l_hv9, l_hv13], 3)

            agg = tf.math.add(concat, concat1)

            agg = concat
            l_1X1 = tfl.Conv2D(in_channel, 1, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                               activation=tf.nn.relu)(agg)

            return l_1X1

        def channel_attention(l1, l2):
            depth = l1.get_shape()[-1]

            op_ch1 = l1
            ch2 = tf.reduce_mean(l2, axis=[1, 2], keep_dims=True)
            ch2 = tfl.Conv2D(depth, 1, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                             activation=None)(ch2)
            ch2 = tf.nn.relu(ch2)
            ch2 = tfl.Conv2D(depth, 1, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                             activation=None)(ch2)
            ch2 = tf.nn.sigmoid(ch2)

            op_ch2 = tf.multiply(ch2, l1)

            ch_op = tf.add(op_ch1, op_ch1)

            return ch_op

        def spatial_attention(l1, l2):
            op_sp1 = l1

            sp2 = tf.reduce_mean(l2, axis=[3], keep_dims=True)
            sp2 = tfl.Conv2D(1, 1, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                             activation=None)(sp2)
            sp2 = tf.nn.relu(sp2)
            sp2 = tfl.Conv2D(1, 1, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                             activation=None)(sp2)
            sp2 = tf.nn.sigmoid(sp2)

            op_sp2 = tf.multiply(sp2, l1)

            sp_op = tf.add(op_sp1, op_sp1)

            return sp_op

        with tf.variable_scope('block_down1', reuse=tf.AUTO_REUSE) as scope:
            l1_res = tfl.Conv2D(8, 3, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                activation=tf.nn.relu)(image)
            l1 = dil_conv(l1_res, 8)
            added_1 = tf.math.add(l1_res, l1)
            l_down1 = tf.space_to_depth(added_1, block_size=2)

        with tf.variable_scope('block_down2', reuse=tf.AUTO_REUSE) as scope:
            l2_res = tfl.Conv2D(32, 3, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                activation=tf.nn.relu)(l_down1)
            l2 = dil_conv(l2_res, 32)
            added_2 = tf.math.add(l2_res, l2)
            l_down2 = tf.space_to_depth(added_2, block_size=2)

        with tf.variable_scope('block_down3', reuse=tf.AUTO_REUSE) as scope:
            l3_res = tfl.Conv2D(128, 3, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                activation=tf.nn.relu)(l_down2)
            l3 = dil_conv(l3_res, 128)
            added_3 = tf.math.add(l3_res, l3)
            l_down3 = tf.space_to_depth(added_3, block_size=2)

        with tf.variable_scope('block_down4', reuse=tf.AUTO_REUSE) as scope:
            l4_res = tfl.Conv2D(512, 3, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                activation=tf.nn.relu)(l_down3)
            l4 = dil_conv(l4_res, 512)
            added_4 = tf.math.add(l4_res, l4)
            l_down4 = tf.space_to_depth(added_4, block_size=2)

        h = l_down4

        with tf.variable_scope('block_up1', reuse=tf.AUTO_REUSE) as scope:
            l_up1 = tf.depth_to_space(h, block_size=2)
            at = tf.space_to_depth(added_1, 8)
            ch_at1 = channel_attention(added_4, at)
            sp_at1 = spatial_attention(added_4, at)
            l_con1 = tf.concat([l_up1, ch_at1, sp_at1], 3)

            ch1 = tf.depth_to_space(l_con1, block_size=8)
            ch1 = tfl.Conv2D(1, 3, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                             activation=tf.nn.relu)(ch1)

        with tf.variable_scope('block_up2', reuse=tf.AUTO_REUSE) as scope:
            l_up2 = tf.depth_to_space(l_con1, block_size=2)
            at = tf.space_to_depth(added_2, 2)
            ch_at2 = channel_attention(added_3, at)
            sp_at2 = spatial_attention(added_3, at)
            l_con2 = tf.concat([l_up2, ch_at2, sp_at2], 3)

            ch2 = tf.depth_to_space(l_con2, block_size=4)
            ch2 = tfl.Conv2D(1, 3, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                             activation=tf.nn.relu)(ch2)

        with tf.variable_scope('block_up3', reuse=tf.AUTO_REUSE) as scope:
            l_up3 = tf.depth_to_space(l_con2, block_size=2)
            at = tf.depth_to_space(added_3, 2)
            ch_at3 = channel_attention(added_2, at)
            sp_at3 = spatial_attention(added_2, at)
            l_con3 = tf.concat([l_up3, ch_at3, sp_at3], 3)

            ch3 = tf.depth_to_space(l_con3, block_size=2)
            ch3 = tfl.Conv2D(1, 3, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                             activation=tf.nn.relu)(ch3)

        with tf.variable_scope('block_up4', reuse=tf.AUTO_REUSE) as scope:
            l_up4 = tf.depth_to_space(l_con3, block_size=2)
            at = tf.depth_to_space(added_4, 8)
            ch_at4 = channel_attention(added_1, at)
            sp_at4 = spatial_attention(added_1, at)
            l_con4 = tf.concat([l_up4, ch_at4, sp_at4], 3)

            ch4 = tfl.Conv2D(1, 3, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                             activation=tf.nn.relu)(l_con4)

        l_out3 = tfl.Conv2D(16, 3, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                            activation=tf.nn.relu)(l_con4)
        l_out2 = tfl.Conv2D(8, 3, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                            activation=tf.nn.relu)(l_out3)
        l_out1 = tfl.Conv2D(1, 3, strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                            activation=tf.nn.relu)(l_out2)

        return l_out1, ch1, ch2, ch3, ch4

def discriminator(x, n_filters=64, k_size=4, scope=None, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope or 'discriminator', reuse=reuse):
        h = conv2d(
            inputs=x,
            num_outputs=n_filters,
            kernel_size=k_size,
            stride=2,
            normalizer_fn=None,
            scope='1')
        h = conv2d(
            inputs=h,
            num_outputs=n_filters * 2,
            kernel_size=k_size,
            stride=2,
            scope='2')
        h = conv2d(
            inputs=h,
            num_outputs=n_filters * 4,
            kernel_size=k_size,
            stride=2,
            scope='3')
        h = conv2d(
            inputs=h,
            num_outputs=n_filters * 8,
            kernel_size=k_size,
            stride=1,
            scope='4')
        h = conv2d(
            inputs=h,
            num_outputs=1,
            kernel_size=k_size,
            stride=1,
            activation_fn=tf.nn.sigmoid,
            scope='5')
        return h

def inhomonet(img_size=192):
    X_real = tf.placeholder(
        name='X', shape=[1, img_size, img_size, 1], dtype=tf.float32)
    Y_real = tf.placeholder(
        name='Y', shape=[1, img_size, img_size, 1], dtype=tf.float32)
    Y_fake_sample = tf.placeholder(
        name='Y_fake_sample',
        shape=[None, img_size, img_size, 1],
        dtype=tf.float32)
    HC = tf.placeholder(
        name='HC', shape=(), dtype=tf.float32)
    DP = tf.placeholder(
        name='3DP', shape=(), dtype=tf.float32)

    Y_fake, ch1, ch2, ch3, ch4 = generator(X_real, scope='G_xy')

    D_Y_real = discriminator(Y_real, scope='D_Y')
    D_Y_fake = discriminator(Y_fake, scope='D_Y', reuse=True)
    D_Y_fake_sample = discriminator(Y_fake_sample, scope='D_Y', reuse=True)

    # Create losses for generators
    l1 = 10
    l2 = 100

    loss_G_xy = l2loss(D_Y_fake, 1.0)
    ssimxy = DSSIMloss(Y_fake, Y_real, max_val=1)
    recon_fy = l1loss(Y_fake, Y_real)
    r1 = l1loss(ch1, Y_real)
    r2 = l1loss(ch2, Y_real)
    r3 = l1loss(ch3, Y_real)
    r4 = l1loss(ch4, Y_real)

    t_r = r1 + r2 + r3 + r4 + recon_fy

    loss_G = loss_G_xy + (l1 * t_r) + HC + DP

    # Create losses for discriminators
    loss_D_Y = l2loss(D_Y_real, 1.0) + l2loss(D_Y_fake_sample, 0.0)

    # Summaries for monitoring training
    tf.summary.image("Y_fake", Y_fake, max_outputs=1)
    tf.summary.image("Y_real", Y_real, max_outputs=1)
    tf.summary.image("X_real", X_real, max_outputs=1)
    tf.summary.image("ch1", ch1, max_outputs=1)
    tf.summary.image("ch2", ch2, max_outputs=1)
    tf.summary.image("ch3", ch3, max_outputs=1)
    tf.summary.image("ch4", ch4, max_outputs=1)

    tf.summary.scalar("Recon_loss", t_r)
    tf.summary.scalar("G_loss", loss_G_xy)
    tf.summary.scalar("His_corr_loss", HC)
    tf.summary.scalar("3DP_loss", DP)
    tf.summary.scalar("Total_loss", loss_G)
    tf.summary.scalar("D_loss", loss_D_Y)

    summaries = tf.summary.merge_all()

    training_vars = tf.trainable_variables()

    D_Y_vars = [v for v in training_vars if v.name.startswith('D_Y')]
    G_xy_vars = [v for v in training_vars if v.name.startswith('G_xy')]

    return locals()


if __name__ == "__main__":
    import os
    print('derp')
    PATH_INHOMONET = '/media/bugger/MyBook/data/pretrained_models/inhomonet/InhomoNet_ABD/inhomonet_ABD'
    sess = tf.compat.v1.Session()
    # restore weights
    tf.compat.v1.disable_eager_execution()
    saver = tf.compat.v1.train.import_meta_graph(os.path.join(PATH_INHOMONET, 'model_final.ckpt.meta'))
#    saver = tf.train.import_meta_graph(os.path.join(PATH_INHOMONET, 'model_final.ckpt.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(PATH_INHOMONET))
    """
    
    
    """
    total_parameters = 0
    np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])

    graph = tf.compat.v1.get_default_graph()
    graph
    Y_real = graph.get_tensor_by_name('Y:0')
    X_real = graph.get_tensor_by_name('X:0')
    Y_fake = graph.get_tensor_by_name('G_xy/conv2d_78/Relu:0')

    #
    import nibabel
    dpatient = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/t2w_n4itk/input'
    ddest = '/media/bugger/MyBook/data/pretrained_models/inhomonet/results/patient_7T'
    files_7T = [os.path.join(dpatient, x) for x in os.listdir(dpatient)]
    patch_shape = (192, 192)
    stride = patch_shape[0] // 4
    import numpy as np
    for sel_file in files_7T:
        file_name = os.path.basename(sel_file).split('.')[0]
        patient_array = nibabel.load(sel_file).get_fdata().T[0, ::-1, ::-1]
        pad_x, pad_y = np.array(patient_array.shape) % patch_shape[0]
        patient_array = np.pad(patient_array, ((0, pad_x), (0, pad_y)))
        import skimage.transform
        import helper.array_transf as harray
        res = harray.get_patches(patient_array, patch_shape=patch_shape, stride=stride)
        res_model = []
        for x_patch in res:
            x_patch = harray.scale_minmax(x_patch)
            x_patch = x_patch[None, :, :, None]
            Y_fake_pred = sess.run([Y_fake], feed_dict={X_real: x_patch})
            res_model.append(Y_fake_pred[0][0, :, :, 0])

        # (patches, target_shape, patch_shape, stride, _dtype=float):
        res_stich = harray.get_stiched(res_model, target_shape=patient_array.shape, patch_shape=patch_shape, stride=stride)
        hplotc.ListPlot([patient_array, res_stich])
        # import matplotlib.pyplot as plt
        # plt.imshow(patient_array[0, :, :, 0])
        # plt.imshow(patient_array[0, :, :, 0])
        z = Y_fake_pred[0][0, :, :, 0]
        import helper.plot_class as hplotc
        hplotc.ListPlot([patient_array[0, :, :, 0], z], col_row=(2,1))
        dfile = os.path.join(ddest, file_name)
        np.save(dfile, z)
        
    # Swithcing to /env/usr/python3..
    import helper.plot_class as hplotc
    import os
    dresult = '/media/bugger/MyBook/data/pretrained_models/inhomonet/results/patient_7T'
    dpng = '/media/bugger/MyBook/data/pretrained_models/inhomonet/results/patient_7T_result_png'
    list_files = [os.path.join(dresult, x) for x in os.listdir(dresult)]
    fig_obj = hplotc.PlotCollage(content_list=list_files, n_display=6, ddest=dpng)
    fig_obj.plot_collage()

    #
    dpatient = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/t2w_n4itk/input'
    ddest = '/media/bugger/MyBook/data/pretrained_models/inhomonet/results/patient_7T_input_png'
    files_7T = [os.path.join(dpatient, x) for x in os.listdir(dpatient)]
    fig_obj = hplotc.PlotCollage(content_list=files_7T, n_display=6, ddest=ddest)
    fig_obj.plot_collage()