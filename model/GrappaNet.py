"""
GrappaNet in Tensorflow..

--> Not very useful if you are going to do non-cartesian stuff..
"""


@tf.function
def model_loss_ssim(y_true, y_pred):
    global lamda
    ssim_loss = 0
    max_val = 1.0
    if tf.reduce_max(y_pred) > 1.0:
        max_val = tf.reduce_max(y_pred)
    ssim_loss = tf.math.abs(
        tf.reduce_mean(tf.image.ssim(img1=y_true, img2=y_pred, max_val=max_val, filter_size=3, filter_sigma=0.1)))
    l1_loss = lamda * tf.reduce_mean(tf.math.abs(y_true - y_pred))
    return 1 - ssim_loss + l1_loss


def conv_block(ip, nfilters, drop_rate):
    layer_top = Conv2D(nfilters, (3, 3), padding="same")(ip)

    # layer_top =BatchNormalization()(layer_top)
    layer_top = tfa.layers.InstanceNormalization(axis=3, center=True,
                                                 scale=True, beta_initializer="random_uniform",
                                                 gamma_initializer="random_uniform")(layer_top)
    res_model = ReLU()(layer_top)
    res_model = Dropout(drop_rate)(res_model)

    res_model = Conv2D(nfilters, (3, 3), padding="same")(res_model)
    res_model = tfa.layers.InstanceNormalization(axis=3, center=True,
                                                 scale=True, beta_initializer="random_uniform",
                                                 gamma_initializer="random_uniform")(res_model)
    # res_model =BatchNormalization()(res_model)
    res_model = Dropout(drop_rate)(res_model)
    res_model = add([layer_top, res_model])
    res_model = ReLU()(res_model)
    return res_model


def encoder(inp, nlayers, nbasefilters, drop_rate):
    skip_layers = []
    layers = inp
    for i in range(nlayers):
        layers = conv_block(layers, nbasefilters * 2 ** i, drop_rate)
        skip_layers.append(layers)
        layers = MaxPooling2D((2, 2))(layers)
    return layers, skip_layers


def decoder(inp, nlayers, nbasefilters, skip_layers, drop_rate):
    layers = inp
    for i in range(nlayers):
        layers = conv_block(layers, nbasefilters * (2 ** (nlayers - 1 - i)), drop_rate)
        layers = UpSampling2D(size=(2, 2), interpolation='bilinear')(layers)
        layers = add([layers, skip_layers.pop()])

    return layers


def create_gen(gen_ip, nlayers, nbasefilters, lambda_l, drop_rate):
    op, skip_layers = encoder(gen_ip, nlayers, nbasefilters, drop_rate)
    op = decoder(op, nlayers, nbasefilters, skip_layers, drop_rate)
    op = Conv2D(18, (3, 3), padding="same")(op)
    return op


def custom_data_consistency(tensors):
    output = tf.where(tf.greater_equal(tensors[0], 1), tensors[0], tensors[1])
    out_cmplx = tf.complex(output[:, :, :, 0:9], output[:, :, :, 9:18])
    ift_sig = tf.signal.fftshift(tf.signal.ifft2d(out_cmplx, name=None))
    real_p = tf.math.real(ift_sig)
    imag_p = tf.math.imag(ift_sig)
    comb = tf.concat(axis=-1, values=[real_p, imag_p])
    # print("here",output.get_shape(),ift_sig.get_shape(),real_p.get_shape(),comb.get_shape(),tensors[0].get_shape())
    return comb


def custom_data_consistency_2(tensors):
    out_cmplx = tf.complex(tensors[1][:, :, :, 0:9], tensors[1][:, :, :, 9:18])
    ft_sig = tf.signal.fftshift(tf.signal.fft2d(out_cmplx, name=None))
    real_p = tf.math.real(ft_sig)
    imag_p = tf.math.imag(ft_sig)
    comb = tf.concat(axis=-1, values=[real_p, imag_p])
    output = tf.where(tf.greater_equal(tensors[0], 1), tensors[0], comb)
    return output


def aux_Grappa_layer(tensor1, tensor2):
    global grappa_wt
    global grappa_p
    t1 = tensor1.numpy()
    t2 = tensor2.numpy()
    # print("max",np.max(t1),np.max(t2),print(t1.shape))

    x_train_cmplx_target = t2[:, :, :, 0:9] + 1j * t2[:, :, :, 9:18]
    x_train_cmplx_target = np.transpose(x_train_cmplx_target, (0, 3, 1, 2))
    l_grappa = []
    for i in range(x_train_cmplx_target.shape[0]):
        res = apply_kernel_weight(kspace=x_train_cmplx_target[i], calib=None,
                                  kernel_size=(5, 5), coil_axis=0,
                                  weights=grappa_wt[int(t1[i][0])], P=grappa_p[int(t1[i][0])])
        res = np.transpose(res, (1, 2, 0))
        out_cmplx_real = tf.convert_to_tensor(res.real)
        out_cmplx_imag = tf.convert_to_tensor(res.imag)
        comb = tf.concat(axis=2, values=[out_cmplx_real, out_cmplx_imag])
        l_grappa.append(comb)
    b_grappa = tf.stack(l_grappa)
    # print("grappa",b_grappa.get_shape())
    return b_grappa


def Grappa_layer(tensor):
    # print(tensor[1].get_shape())
    out_tensor = tf.py_function(func=aux_Grappa_layer, inp=tensor, Tout=tf.float32)
    out_tensor.set_shape(tensor[1].get_shape())
    # print(out_tensor.get_shape())
    return out_tensor


def ift_RSS(tensor):
    cmplx_tensor = tf.complex(tensor[:, :, :, 0:9], tensor[:, :, :, 9:18])
    ift_sig = tf.signal.fftshift(tf.signal.ifft2d(cmplx_tensor, name=None))
    Y_rss = tf.math.sqrt(tf.math.reduce_sum(tf.square(tf.math.abs(ift_sig)), axis=3))

    return Y_rss

def build_model(input_shape, n_filter=32, n_depth=4, lamda=1, dropout_rate=0.05):
    # first pass
    input_layer = Input(shape=input_shape)
    input_layer_grappa_wt_indx = Input(shape=(1))
    kspace_u1 = create_gen(input_layer, n_depth, n_filter, lamda, dropout_rate)
    data_con_layer = Lambda(custom_data_consistency, name="data_const_K_u1")([input_layer, kspace_u1])
    img_space_u1 = create_gen(data_con_layer, n_depth, n_filter, lamda, dropout_rate)
    data_con_layer = Lambda(custom_data_consistency_2, name="data_const_K_u1_2")([input_layer, img_space_u1])
    grappa_recon_k = Lambda(Grappa_layer, name="data_const_K_2")([input_layer_grappa_wt_indx, data_con_layer])

    # second Pass
    kspace_u2 = create_gen(grappa_recon_k, n_depth, n_filter, lamda, dropout_rate)
    data_con_layer = Lambda(custom_data_consistency, name="data_const_K_u2")([input_layer, kspace_u2])
    img_space_u2 = create_gen(data_con_layer, n_depth, n_filter, lamda, dropout_rate)
    data_con_layer = Lambda(custom_data_consistency_2, name="data_const_K_u2_2")([input_layer, img_space_u2])

    # IFT+RSS
    data_con_layer = Lambda(ift_RSS, name="IFT_RSS")(data_con_layer)
    return Model(inputs=[input_layer, input_layer_grappa_wt_indx], outputs=data_con_layer)