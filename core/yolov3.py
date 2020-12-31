#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import core.common as common
import core.backbone as backbone

def YOLOv3(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.darknet53(input_layer)

    conv = common.conv_bn_relu(conv, (1, 1, 1024,  512))
    conv = common.conv_bn_relu(conv, (3, 3,  512, 1024))
    conv = common.conv_bn_relu(conv, (1, 1, 1024,  512))
    conv = common.conv_bn_relu(conv, (3, 3,  512, 1024))
    conv = common.conv_bn_relu(conv, (1, 1, 1024,  512))

    conv_lobj_branch = common.conv_bn_relu(conv, (3, 3, 512, 1024))
    conv_lbbox = common.conv_bn_relu(conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.conv_bn_relu(conv, (1, 1,  512,  256))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.conv_bn_relu(conv, (1, 1, 768, 256))
    conv = common.conv_bn_relu(conv, (3, 3, 256, 512))
    conv = common.conv_bn_relu(conv, (1, 1, 512, 256))
    conv = common.conv_bn_relu(conv, (3, 3, 256, 512))
    conv = common.conv_bn_relu(conv, (1, 1, 512, 256))

    conv_mobj_branch = common.conv_bn_relu(conv, (3, 3, 256, 512))
    conv_mbbox = common.conv_bn_relu(conv_mobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.conv_bn_relu(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.conv_bn_relu(conv, (1, 1, 384, 128))
    conv = common.conv_bn_relu(conv, (3, 3, 128, 256))
    conv = common.conv_bn_relu(conv, (1, 1, 256, 128))
    conv = common.conv_bn_relu(conv, (3, 3, 128, 256))
    conv = common.conv_bn_relu(conv, (1, 1, 256, 128))

    conv_sobj_branch = common.conv_bn_relu(conv, (3, 3, 128, 256))
    conv_sbbox = common.conv_bn_relu(conv_sobj_branch, (1, 1, 256, 3*(NUM_CLASS +5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def tiny_YOLOv3_modify(input_layer, NUM_CLASS):
    route_1, conv = backbone.darknet53_tiny(input_layer)

    conv = common.conv_bn_relu(conv, (1, 1, 1024, 512))

    conv_lobj_branch = common.conv_bn_relu(conv, (3, 3, 512, 512))
    conv_lbbox = common.conv_bn_relu(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.conv_bn_relu(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)
    # conv = tf.keras.layers.concatenate([conv, route_1], axis=-1)

    conv_mobj_branch = common.conv_bn_relu(conv, (3, 3, 256+256, 386))
    conv_mbbox = common.conv_bn_relu(conv_mobj_branch, (1, 1, 386, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def tiny_YOLOv3_add(input_layer, NUM_CLASS):
    route_1, conv = backbone.darknet53_tiny(input_layer)

    conv = common.conv_bn_relu(conv, (1, 1, 1024, 512))

    conv_lobj_branch = common.conv_bn_relu(conv, (3, 3, 512, 512))
    conv_lbbox = common.conv_bn_relu(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.conv_bn_relu(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    # conv = tf.concat([conv, route_1], axis=-1)
    conv = conv + route_1
    conv_mobj_branch = common.conv_bn_relu(conv, (3, 3, 256, 256))
    conv_mbbox = common.conv_bn_relu(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def tiny_YOLOv3(input_layer, NUM_CLASS):
    route_1, conv = backbone.darknet53_tiny(input_layer)

    conv = common.conv_bn_relu(conv, (1, 1, 1024, 256))

    conv_lobj_branch = common.conv_bn_relu(conv, (3, 3, 256, 512))
    conv_lbbox = common.conv_bn_relu(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.conv_bn_relu(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    # conv = tf.concat([conv, route_1], axis=-1)
    conv = tf.keras.layers.concatenate([conv, route_1], axis=-1)

    conv_mobj_branch = common.conv_bn_relu(conv, (3, 3, 128, 256))
    conv_mbbox = common.conv_bn_relu(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def mobilenetV2_YOLOv3(input_layer, NUM_CLASS):
    route_1, conv = backbone.mobilenet_v2(input_layer)

    conv = common.conv_bn_relu(conv, (1, 1, 320, 512))     #mobilenet中的输出的通道数太少，进行升通道
    route_1 = common.conv_bn_relu(route_1, (1, 1, 96, 256))

    conv_lobj_branch = common.conv_bn_relu(conv, (3, 3, 512, 512))
    conv_lbbox = common.conv_bn_relu(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.conv_bn_relu(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.conv_bn_relu(conv, (3, 3, 256+256, 386))
    conv_mbbox = common.conv_bn_relu(conv_mobj_branch, (1, 1, 386, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]


def peleenet_YOLOV3(input_layer, NUM_CLASS):
    # 如果使用stem_block，这stage3_tb和stege4_tb的stride为32，如果不使用则为8
    stage3_tb, stage4_tb = backbone.peleeNet(input_layer)
    x = common.conv_bn_relu(stage4_tb, (1, 1, 256), prefix='ext/fe1_1')
    fe2_1 = common.conv_bn_relu(x, (3, 3, 256), downsample=True, prefix='ext/fe1_2')
    x = common.conv_bn_relu(fe2_1, (1, 1, 256), prefix='ext/fe2_1')
    fe2_2 = common.conv_bn_relu(x, (3, 3, 256), downsample=True, prefix='ext/fe2_2')


    out1 = common.peleenet_residual_block(stage3_tb, 256, prefix='stage3_tb/rb')
    out2 = common.peleenet_residual_block(fe2_1, 256, prefix='ext/fe1_1/rb')
    out3 = common.peleenet_residual_block(fe2_2, 256, prefix='ext/fe2_1/rb')

    conv_sobj_branch = common.conv_bn_relu(out1, (3, 3, 256), prefix='s_branch')
    conv_sbbox = common.conv_bn_relu(conv_sobj_branch, (1, 1, 3*(NUM_CLASS +5)), activate=False, bn=False, prefix='s_obj')

    conv_mobj_branch = common.conv_bn_relu(out2, (3, 3, 256), prefix='m_branch')
    conv_mbbox = common.conv_bn_relu(conv_mobj_branch, (1, 1, 3 * (NUM_CLASS + 5)), activate=False, bn=False, prefix='m_obj')

    conv_lobj_branch = common.conv_bn_relu(out3, (3, 3, 256), prefix='l_branch')
    conv_lbbox = common.conv_bn_relu(conv_lobj_branch, (1, 1, 3 * (NUM_CLASS + 5)), activate=False, bn=False, prefix='l_obj')
    return  conv_sbbox, conv_mbbox, conv_lbbox