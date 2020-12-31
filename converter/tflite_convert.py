# coding=utf-8
import os, glob
import numpy as np
import cv2
import tensorflow as tf
from absl import flags, app
from absl.flags import FLAGS
import random
from converter.utils import  get_graph
from core.utils import image_preprocess

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
# if gpus:
#     tf.config.experimental.set_visible_devices(devices=gpus[-1], device_type='GPU')
    # tf.config.experimental.set_memory_growth(device=gpus[1], enable=True)

# target = 'helmet'
# target = 'pedestrians'
target = 'head'

# backbone = 'mobilenetv2'
# backbone = 'tiny-yolov3'
# backbone = 'yolov3'
backbone = 'peleenet'

flags.DEFINE_string('quantization_type',                  'IO',              'DR for dynamic_range,  IO for integer_only, NQ for not quantization, FI for full integer')
flags.DEFINE_list('tensor_type',             [tf.uint8, tf.float32],            'fp32 or uint8 for input and output tensor date type')

if target == 'helmet':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    flags.DEFINE_string('quan_images',  'data/quan/helmet',                     'images to quan')

    flags.DEFINE_string('anchors_file', 'data/anchors/helmet_540_960_6_anchors.txt',     'anchors file')
    flags.DEFINE_string('classes_file', 'data/classes/helmet.names',     'anchors file')
    if backbone == 'mobilenetv2':
        flags.DEFINE_string('ckpt', 'save/helmet/ckpt/mobilenetv2_yolov3_540_960/model_final','path to weights file')
        flags.DEFINE_string('tflitePrefix', 'save/helmet/helmet_mobilenetv2_yolov3_540_960','path to save tflite')
    elif backbone == 'tiny-yolov3':
        flags.DEFINE_string('ckpt', 'save/helmet/ckpt/tiny_yolov3_540_960/model_epoch50', 'path to weights file')
        flags.DEFINE_string('tflitePrefix', 'save/helmet/helmet_tiny_yolov3_540_960_test', 'path to save tflite')

elif target == 'pedestrians':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    # flags.DEFINE_string('anchors_file', 'data/anchors/pedestrian_540_960_6_anchors.txt',     'anchors file')
    # flags.DEFINE_string('quan_images',  'data/quan/pedestrian1',                     'images to quan')
    flags.DEFINE_string('anchors_file', 'data/anchors/person_crowdhuman_540_960_9_anchors.txt',     'anchors file')
    flags.DEFINE_string('quan_images',  'data/quan/head',                     'images to quan')
    flags.DEFINE_string('classes_file', 'data/classes/pedestrian.names',     'classes file')
    if backbone == 'mobilenetv2':
        flags.DEFINE_string('ckpt',         'save/pedestrian/ckpt/mobilenetv2_yolov3_540_960/model_final',  'path to weights file')
        flags.DEFINE_string('tflitePrefix', 'save/pedestrian/pedestrians_mobilenetv2_yolov3_540_960',          'path to save tflite')
    elif backbone == 'tiny-yolov3':
        flags.DEFINE_string('ckpt',         'save/pedestrian/ckpt/tiny_yolov3_540_960/model_final',  'path to weights file')
        flags.DEFINE_string('tflitePrefix', 'save/pedestrian/pedestrians_tiny_yolov3_540_960',          'path to save tflite')
    elif backbone == 'yolov3':
        flags.DEFINE_string('ckpt',         'save/pedestrian/ckpt/yolov3_480_640/model_epoch42',  'path to weights file')
        flags.DEFINE_string('tflitePrefix', 'save/pedestrian/pedestrians_yolov3_480_640',          'path to save tflite')
        flags.DEFINE_string('savePrefix',   'save/pedestrian/savedmodel/yolov3_480_640.h5',          'path to savedmodel')
    elif backbone == 'peleenet':
        flags.DEFINE_string('ckpt', 'save/pedestrian/ckpt/peleenet_yolov3_540_960/model_epoch10', 'path to weights file')
        flags.DEFINE_string('tflitePrefix', 'save/pedestrian/pedestrians_peleenet_yolov3_540_960', 'path to save tflite')

elif target == 'head':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    flags.DEFINE_string('quan_images',  'data/quan/head',                     'images to quan')
    flags.DEFINE_string('anchors_file', 'data/anchors/head_crowdhuman_540_960_9_anchors.txt',     'anchors file')
    flags.DEFINE_string('classes_file', 'data/classes/head.names',     'classes file')
    if backbone == 'peleenet':
        flags.DEFINE_string('ckpt', 'save/head/ckpt/peleenet_yolov3_540_960/model_final', 'path to weights file')
        flags.DEFINE_string('tflitePrefix', 'save/head/head_peleenet_yolov3_540_960', 'path to save tflite')

def representative_dataset_gen_img(imgCnt=0):
    imgDir = os.path.abspath(FLAGS.quan_images)
    imgNameList = glob.glob(os.path.join(imgDir, '*.jpg'))
    imgIndex = random.sample(range(len(imgNameList)), len(imgNameList) if imgCnt == 0 else imgCnt)
    for i in imgIndex:
        print(imgNameList[i])
        imgMat = cv2.imread(imgNameList[i])
        imgMat = image_preprocess(np.copy(imgMat), [FLAGS.input_size[0], FLAGS.input_size[1]])
        imgMat = imgMat[np.newaxis,:,:,:]
        yield [tf.constant(imgMat.astype(np.float32))]

def convert_to_tflite(converter, TFLitePrefix):
    if FLAGS.quantization_type == 'DR':
        # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE] #for Dynamic range quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT] #到底是哪个才对
        TFLiteFile = TFLitePrefix + '_DR.tflite'
    elif FLAGS.quantization_type == 'FI':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen_img
        TFLiteFile = TFLitePrefix + '_FI.tflite'
    elif FLAGS.quantization_type == 'IO':
        # converter.allow_custom_ops = True
        # This enables quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # This ensures that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        #当有tflite没有实现的op时，用tf自身的实现方式
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
        converter.target_spec.supported_types = [tf.int8]
        # These set the input and output tensors to uint8 (added in r2.3)
        converter.inference_input_type = FLAGS.tensor_type[0]
        converter.inference_output_type = FLAGS.tensor_type[1]
        # This sets the representative dataset for quantization
        converter.representative_dataset = representative_dataset_gen_img
        TFLiteFile = TFLitePrefix + '_IO.tflite'
    else:
        print("No quantization.")
        TFLiteFile = TFLitePrefix + '.tflite'
    TFLiteModel = converter.convert()
    print("Convert to tflite succeed:", TFLiteFile)
    open(TFLiteFile, "wb").write(TFLiteModel)

'''
这个地方直接使用TF2.2中的TOCO进行量化，输入和输出为了兼容多个平台，就算指定了量化为int8，最后也只会是fp32.
在TF2.3及以上的版本，量化工具升级为MLIR，这个时候输入和输出也会量化为int8.
现在暂时使用v1中的版本能满足输入输出量化为int8.
'''
def convert_tflite_from_savedmodel(TFLiteFile, model, savedModelDir=None):
    if savedModelDir is None:
        savedModelDir = "save/tmp/savedmodel"
        model.save(savedModelDir)
    converter = tf.lite.TFLiteConverter.from_saved_model(savedModelDir)
    convert_to_tflite(converter, TFLiteFile)

def convert_tflite_from_keras(TFLiteFile, model, kerasH5File=None):
    if kerasH5File is None:
        kerasH5File = "save/tmp/tmp.h5"
        model.save(kerasH5File)
        converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(kerasH5File)
        # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    convert_to_tflite(converter, TFLiteFile)

def main(_argv):
    input_tensor = tf.keras.layers.Input([FLAGS.input_size[0], FLAGS.input_size[1], 3], dtype=tf.float32)
    graph = get_graph(input_tensor, FLAGS.classes_file, FLAGS.anchors_file, backbone, withDecode=False)
    model = tf.keras.Model(input_tensor, graph)
    model.load_weights(FLAGS.ckpt)
    # model.load_weights(tf.train.latest_checkpoint(FLAGS.ckpt))

    convert_tflite_from_savedmodel(FLAGS.tflitePrefix, model)      #转换得到的tflite的input和output为fp32
    # convert_tflite_from_keras(FLAGS.tflitePrefix, model)             #转换得到的tflite input和output为uint8


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


