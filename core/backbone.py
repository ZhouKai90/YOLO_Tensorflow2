#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-07-11 23:37:51
#   Description :
#
#================================================================

import tensorflow as tf
import core.common as common

def darknet53(input_data):

    input_data = common.convolutional(input_data, (3, 3,  3,  32))
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64)

    input_data = common.convolutional(input_data, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        input_data = common.residual_block(input_data, 128,  64, 128)

    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = common.residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data

def darknet53_tiny(input_data):
    input_data = common.convolutional(input_data, (3, 3, 3, 16))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 16, 32))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 32, 64))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 64, 128))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 128, 256))
    route_1 = input_data
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 256, 512))
    input_data = tf.keras.layers.MaxPool2D(2, 1, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))

    return route_1, input_data

def mobilenet_v2(input_data, branch_size, alpha=1.0):
    first_block_filters = common.make_divisible(32 * alpha, 8)
    # Group1 stride=2
    x = common.convolutional(input_data, (3, 3, 3, first_block_filters),
                             downsample=True, activate_type='relu6')
    # Group2 stride=2
    x = common.inverted_res_block(
        x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)
    # Group3 stride=4
    x = common.inverted_res_block(
        x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    x = common.inverted_res_block(
        x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)
    # Group4 stride=8
    x = common.inverted_res_block(
        x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = common.inverted_res_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = common.inverted_res_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)
    small_pred = x
    # Group5 stride=16
    x = common.inverted_res_block(
        x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    x = common.inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    x = common.inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    x = common.inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)
    # Group6 stride=16
    x = common.inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    x = common.inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    x = common.inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)
    medium_pred = x
    # Group7 stride=32
    x = common.inverted_res_block(
        x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
    x = common.inverted_res_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    x = common.inverted_res_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)
    # Group8 stride=32
    x = common.inverted_res_block(
        x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)
    large_pred = x
    if branch_size == 2:
        return medium_pred, large_pred
    elif branch_size == 3:
        return small_pred, medium_pred, large_pred