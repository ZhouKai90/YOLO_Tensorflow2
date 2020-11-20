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

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='relu'):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = tf.keras.layers.LeakyReLU(conv, alpha=0.1)(conv)
            # conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "relu":
            conv = tf.keras.layers.ReLU()(conv)
            # conv = tf.nn.relu(conv)
        elif activate_type == "relu6":
            conv = tf.keras.layers.ReLU(6)(conv)
            # conv = tf.nn.relu6(conv)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)

def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2))

    residual_output = short_cut + conv
    return residual_output

def upsample(input_layer, method="resize"):
    assert method in ["resize", "deconv"]
    if method == 'resize':
        # return tf.image.resize(input_layer, [input_layer.shape[1] * 2, input_layer.shape[2] * 2], method='bilinear')
        return tf.image.resize(input_layer, [input_layer.shape[1] * 2, input_layer.shape[2] * 2], method='nearest')
    if method == 'deconv':
        numm_filter = input_layer.shape.as_list()[-1]
        # return tf.keras.layers.Conv2DTranspose(numm_filter, kernel_size=2, padding='same',strides=(2, 2))(input_layer)
        return tf.keras.layers.UpSampling2D(size=(2, 2))(input_layer)

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
