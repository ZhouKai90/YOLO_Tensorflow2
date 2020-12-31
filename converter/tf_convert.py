# coding=utf-8
import os
import tensorflow as tf
from absl import flags, app
from absl.flags import FLAGS
from converter.utils import get_graph

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
# if gpus:
#     tf.config.experimental.set_visible_devices(devices=gpus[-1], device_type='GPU')
    # tf.config.experimental.set_memory_growth(device=gpus[1], enable=True)

# target = 'helmet'
target = 'pedestrians'
# target = 'head'

# backbone = 'mobilenetv2'
# backbone = 'tiny-yolov3'
# backbone = 'yolov3'
backbone = 'peleenet'

if target == 'helmet':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    flags.DEFINE_string('anchors_file', 'data/anchors/helmet_540_960_6_anchors.txt',     'anchors file')
    flags.DEFINE_string('classes_file', 'data/classes/helmet.names',     'anchors file')
    if backbone == 'mobilenetv2':
        flags.DEFINE_string('ckpt', 'save/helmet/ckpt/mobilenetv2_yolov3_540_960/model_final','path to weights file')
        flags.DEFINE_string('savePrefix', 'save/helmet/savedmodel/mobilenetv2_yolov3_540_960.h5','path to savedmodel')
    elif backbone == 'tiny-yolov3':
        flags.DEFINE_string('ckpt', 'save/helmet/ckpt/tiny_yolov3_540_960/model_epoch50', 'path to weights file')
        flags.DEFINE_string('savePrefix', 'save/helmet/savedmodel/tiny_yolov3_540_960', 'path to savedmodel')

elif target == 'pedestrians':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    # flags.DEFINE_string('anchors_file', 'data/anchors/pedestrian_540_960_6_anchors.txt',     'anchors file')
    flags.DEFINE_string('anchors_file', 'data/anchors/person_crowdhuman_540_960_9_anchors.txt',     'anchors file')
    flags.DEFINE_string('classes_file', 'data/classes/pedestrian.names',     'classes file')
    if backbone == 'mobilenetv2':
        flags.DEFINE_string('ckpt',         'save/pedestrian/ckpt/mobilenetv2_yolov3_540_960/model_final',  'path to weights file')
        flags.DEFINE_string('savePrefix',   'save/pedestrian/savedmodel/mobilenetv2_yolov3_540_960',          'path to savedmodel')
    elif backbone == 'tiny-yolov3':
        flags.DEFINE_string('ckpt',         'save/pedestrian/ckpt/tiny_yolov3_540_960/model_final',  'path to weights file')
        flags.DEFINE_string('savePrefix',   'save/pedestrian/savedmodel/tiny_yolov3_540_960',          'path to savedmodel')
    elif backbone == 'yolov3':
        flags.DEFINE_string('ckpt',         'save/pedestrian/ckpt/yolov3_480_640/model_epoch42',  'path to weights file')
        flags.DEFINE_string('savePrefix',   'save/pedestrian/savedmodel/yolov3_480_640.h5',          'path to savedmodel')
    elif backbone == 'peleenet':
        flags.DEFINE_string('ckpt', 'save/pedestrian/ckpt/peleenet_yolov3_540_960/model_final', 'path to weights file')
        flags.DEFINE_string('savePrefix', 'save/pedestrian/savedmodel/peleenet_yolov3_540_960', 'path to savedmodel')

elif target == 'head':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    flags.DEFINE_string('anchors_file', 'data/anchors/head_crowdhuman_540_960_9_anchors.txt',     'anchors file')
    flags.DEFINE_string('classes_file', 'data/classes/head.names',     'classes file')
    if backbone == 'peleenet':
        flags.DEFINE_string('ckpt', 'save/head/ckpt/peleenet_yolov3_540_960/model_final', 'path to weights file')
        flags.DEFINE_string('savePrefix', 'save/head/savedmodel/peleenet_yolov3_540_960', 'path to savedmodel')


def main(_argv):
    input_tensor = tf.keras.layers.Input([FLAGS.input_size[0], FLAGS.input_size[1], 3], dtype=tf.float32)
    graph = get_graph(input_tensor, FLAGS.classes_file, FLAGS.anchors_file, backbone, withDecode=True)
    model = tf.keras.Model(input_tensor, graph)
    model.load_weights(FLAGS.ckpt)
    model.save(FLAGS.savePrefix)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
