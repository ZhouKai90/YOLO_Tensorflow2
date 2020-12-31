#! /usr/bin/env python
# coding=utf-8
import tensorflow as tf
from tensorflow.python.keras import backend

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def conv_bn_relu(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='relu6', prefix=None):
    if downsample:
        # name = None if prefix == None else prefix + '_pad'
        # input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)), name=name)(input_layer)
        # padding = 'valid'
        padding = 'same'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    name = None if prefix == None else prefix + '_conv2d'
    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.),
                                  name=name)(input_layer)

    if bn:
        name = None if prefix == None else prefix + '_bn'
        conv = BatchNormalization(name=name)(conv)
    if activate:
        if activate_type == "leaky":
            name = None if prefix == None else prefix + '_leaky_relu'
            conv = tf.keras.layers.LeakyReLU(conv, alpha=0.1, name=name)(conv)
        elif activate_type == "relu":
            name = None if prefix == None else prefix + '_relu'
            conv = tf.keras.layers.ReLU(name=name)(conv)
        elif activate_type == "relu6":
            name = None if prefix == None else prefix + '_relu6'
            conv = tf.keras.layers.ReLU(6, name=name)(conv)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='relu6'):
    short_cut = input_layer
    conv = conv_bn_relu(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = conv_bn_relu(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output

def peleenet_residual_block(input_layer, num_filter, bottleneck_fact=0.5, prefix=None):
    branch1 = conv_bn_relu(input_layer, (1, 1, num_filter), activate=False, prefix=prefix+'/b1')
    branch2a = conv_bn_relu(branch1, (1, 1, int(num_filter*bottleneck_fact)), prefix=prefix+'/b2a')
    branch2b = conv_bn_relu(branch2a, (3, 3, int(num_filter*bottleneck_fact)), prefix=prefix+'/b2b')
    branch2c = conv_bn_relu(branch2b, (1, 1, num_filter), activate=False, prefix=prefix+'/b2c')
    residual_output = branch1 + branch2c
    return tf.keras.layers.ReLU(6, name=prefix+'/relu6')(residual_output)



def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

def upsample(input_layer, method="resize"):
    assert method in ["resize", "deconv"]
    if method == 'resize':
        return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')
        # return tf.image.resize(input_layer, [input_layer.shape[1] * 2, input_layer.shape[2] * 2], method='bilinear')
    if method == 'deconv':
        numm_filter = input_layer.shape.as_list()[-1]
        return tf.keras.layers.Conv2DTranspose(numm_filter, kernel_size=2, padding='valid',strides=(2, 2))(input_layer)
        # return tf.keras.layers.Conv2DTranspose(numm_filter, kernel_size=2, padding='same',strides=(2, 2))(input_layer)
        # return tf.keras.layers.UpSampling2D(size=(2, 2))(input_layer)

def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Arguments:
    inputs: Input tensor.
    kernel_size: An integer or tuple/list of 2 integers.

    Returns:
    A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]),
          (correct[1] - adjust[1], correct[1]))

def inverted_res_block(inputs, expansion, filters, stride, alpha, block_id):
    """Inverted ResNet block."""
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = make_divisible(pointwise_conv_filters, 8)

    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = tf.keras.layers.Conv2D(
            expansion * in_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None,
            name=prefix + 'expand')(
                x)
        x = tf.keras.layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand_BN')(
                x)
        x = tf.keras.layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

  # Depthwise
    if stride == 2:
        # x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)), name=prefix + 'pad')(x)
        x = tf.keras.layers.ZeroPadding2D(
            padding=correct_pad(x, 3),
            name=prefix + 'pad')(x)

    x = tf.keras.layers.DepthwiseConv2D(
      kernel_size=3,
      strides=stride,
      activation=None,
      use_bias=False,
      padding='same' if stride == 1 else 'valid',
      name=prefix + 'depthwise')(
          x)
    x = tf.keras.layers.BatchNormalization(
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'depthwise_BN')(
          x)

    x = tf.keras.layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = tf.keras.layers.Conv2D(
      pointwise_filters,
      kernel_size=1,
      padding='same',
      use_bias=False,
      activation=None,
      name=prefix + 'project')(
          x)
    x = tf.keras.layers.BatchNormalization(
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'project_BN')(
          x)

    if in_channels == pointwise_filters and stride == 1:
        return tf.keras.layers.Add(name=prefix + 'add')([inputs, x])
    return x

def stem_block(input_tensor, downsample=False):
    if downsample:
        x = conv_bn_relu(input_tensor, (3, 3, 32), downsample=True, prefix='stm1')
    else:
        x=input_tensor
    branch1 = conv_bn_relu(x, (1, 1, 16), prefix='stm2a')
    branch1 = conv_bn_relu(branch1, (3, 3, 32), downsample=True, prefix='stm2b')
    branch2 = tf.keras.layers.MaxPool2D(2, name='stm/pool')(x)
    x = tf.keras.layers.Concatenate(name='stm/concat')([branch1, branch2])
    x = conv_bn_relu(x, (1, 1, 32), prefix='stm3')
    return x

def dense_block(input_tensor, num_layers, growth_rate, bottleneck_width, db_prefix):
    x = input_tensor
    growth_rate = int(growth_rate / 2)

    for i in range(num_layers):
        inter_channel = int(growth_rate*bottleneck_width/4) * 4
        branch1 = conv_bn_relu(x, (1, 1, inter_channel), prefix=db_prefix+'_{}/branch1a'.format(i+1))
        branch1 = conv_bn_relu(branch1, (3, 3, growth_rate), prefix=db_prefix+'_{}/branch1b'.format(i+1))

        branch2 = conv_bn_relu(x, (1, 1, inter_channel), prefix=db_prefix+'_{}/branch2a'.format(i+1))
        branch2 = conv_bn_relu(branch2, (3, 3, growth_rate), prefix=db_prefix+'_{}/branch2b'.format(i+1))
        branch2 = conv_bn_relu(branch2, (3, 3, growth_rate), prefix=db_prefix+'_{}/branch2c'.format(i+1))
        x = tf.keras.layers.Concatenate(name=db_prefix+'_{}/concat'.format(i+1))([x, branch1, branch2])
    return x

def transition_layer(input_tensor, k, use_pooling=True, prefix=None):
    x = conv_bn_relu(input_tensor, (1, 1, k), prefix=prefix)
    if use_pooling:
        return tf.keras.layers.AveragePooling2D(2, name=prefix+'/pool', padding='same')(x)
    else:
        return x