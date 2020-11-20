import os, glob
import numpy as np
import cv2
import tensorflow as tf
from core.yolov3 import YOLOv3, tiny_YOLOv3, mobilenetV2_YOLOv3
from core.model_factory import decode
from absl import flags, app
from absl.flags import FLAGS
import core.utils as utils

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
# if gpus:
#     tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
    # tf.config.experimental.set_memory_growth(device=gpus[1], enable=True)

# target = 'helmet'
target = 'pedestrians'

backbone = 'mobilenetv2'
# backbone = 'tiny'

flags.DEFINE_integer('branch_size',              2,               'branch size')
flags.DEFINE_string('framework',                'tflite',          'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_string('quantization_type',        'uint8',            'dynamic_range or float16 or uint8 or None')

if target == 'helmet':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    flags.DEFINE_string('quan_images',  'data/quan/helmet',                     'images to quan')
    flags.DEFINE_string('anchors_file', 'data/anchors/helmet_540_960_6_anchors.txt',     'anchors file')
    flags.DEFINE_string('classes_file', 'data/classes/helmet.names',     'anchors file')

    flags.DEFINE_string('ckpt',         'save/helmet/ckpt/tiny_yolov3_540_960/model_final',  'path to weights file')
    flags.DEFINE_string('tflite',       'save/helmet/tiny_yolov3_helmet540_960_fp32.tflite',          'path to save tflite')
    flags.DEFINE_string('savePrefix',   'save/helmet/savedmodel/tiny_yolov3_540_960',          'path to savedmodel')
elif target == 'pedestrians':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    flags.DEFINE_string('quan_images',  'data/quan/pedestrian',                     'images to quan')
    flags.DEFINE_string('anchors_file', 'data/anchors/pedestrian_540_960_6_anchors.txt',     'anchors file')
    flags.DEFINE_string('classes_file', 'data/classes/pedestrian.names',     'classes file')
    if backbone == 'mobilenetv2':
        flags.DEFINE_string('ckpt',         'save/pedestrian/ckpt/mobilenetv2_yolov3_540_960/model_final',  'path to weights file')
        flags.DEFINE_string('tflite',       'save/pedestrian/pedestrians_mobilenetv2_yolov3_540_960_uint8.tflite',          'path to save tflite')
        flags.DEFINE_string('savePrefix',   'save/pedestrian/savedmodel/mobilenetv2_yolov3_540_960',          'path to savedmodel')
    elif backbone == 'tiny':
        flags.DEFINE_string('ckpt',         'save/pedestrian/ckpt/tiny_yolov3_540_960_1W/model_final',  'path to weights file')
        flags.DEFINE_string('tflite',       'save/pedestrian/tiny_yolov3_pedstrian_540_960_int8_epoch50.tflite',          'path to save tflite')
        flags.DEFINE_string('savePrefix',   'save/pedestrian/savedmodel/tiny_yolov3_540_960_1W.h5',          'path to savedmodel')


def getModelformCheckpoint(withDecode=False, swap=False):
    numClasses = len(utils.read_class_names(FLAGS.classes_file))
    anchors = utils.get_anchors(FLAGS.anchors_file, FLAGS.branch_size)
    input_tensor = tf.keras.layers.Input([FLAGS.input_size[0], FLAGS.input_size[1], 3])
    if backbone == 'yolov3':
        feature_maps = YOLOv3(input_tensor, numClasses)
    elif backbone == 'tiny':
        feature_maps = tiny_YOLOv3(input_tensor, numClasses)
    elif backbone == 'mobilenetv2':
        feature_maps = mobilenetV2_YOLOv3(input_tensor, numClasses, FLAGS.branch_size)
    else:
        raise ValueError('backbone unknow.')

    bboxTensors = []
    if withDecode:
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode(fm, i, anchors, numClasses)
            bboxTensors.append(bbox_tensor)
    else:
        if swap:
            for l in feature_maps:
                out = tf.transpose(l, [0, 3, 1, 2])
                bboxTensors.append(out)
        else:
            bboxTensors = feature_maps

    model = tf.keras.Model(input_tensor, bboxTensors)
    model.load_weights(FLAGS.ckpt)
    return model

def representative_dataset_gen():
    for _ in range(20):
        print('range')
        yield [np.random.uniform(0.0, 1.0, size=(1, FLAGS.input_size[0], FLAGS.input_size[1], 3)).astype(np.float32)]

def representative_dataset_gen_img():
    imgDir = os.path.abspath(FLAGS.quan_images)
    imgNameList = glob.glob(os.path.join(imgDir, '*.jpg'))
    for img in imgNameList:
        print(img)
        imgMat = cv2.imread(img)
        imgMat = utils.image_preporcess(np.copy(imgMat), [FLAGS.input_size[0], FLAGS.input_size[1]])
        imgMat = imgMat[np.newaxis,:,:,:]
        yield [imgMat.astype(np.float32)]

def convert_to_tflite(converter, TFLiteFile):
    if FLAGS.quantization_type == 'dynamic_range':
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE] #for Dynamic range quantization
        converter.representative_dataset = representative_dataset_gen_img
    elif FLAGS.quantization_type == 'uint8':
        # converter.allow_custom_ops = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        # converter.representative_dataset = representative_dataset_gen
        converter.representative_dataset = representative_dataset_gen_img
    else:
        print("No quantization.")
    TFLiteModel = converter.convert()
    print("Convert to tflite succeed:", TFLiteFile)
    open(TFLiteFile, "wb").write(TFLiteModel)

'''
这个地方直接使用TF2.2中的TOCO进行量化，输入和输出为了兼容多个平台，就算指定了量化为int8，最后也只会是fp32.
在TF2.3及以上的版本，量化工具升级为MLIR，这个时候输入和输出也会量化为int8.
现在暂时使用v1中的版本能满足输入输出量化为int8.
'''
def convert_tflite_from_savedmodel(TFLiteFile, savedModelDir=None):
    if savedModelDir is None:
        savedModelDir = "save/tmp/savedmodel"
        model = getModelformCheckpoint()
        model.save(savedModelDir)
    # converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(savedModelDir)
    converter = tf.lite.TFLiteConverter.from_saved_model(savedModelDir)
    convert_to_tflite(converter, TFLiteFile)

def convert_tflite_from_keras(TFLiteFile, kerasH5File=None):
    if kerasH5File is None:
        kerasH5File = "save/tmp/tmp.h5"
        model = getModelformCheckpoint()
        model.save(kerasH5File)
        converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(kerasH5File)
        # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    convert_to_tflite(converter, TFLiteFile)

def main(_argv):
    if FLAGS.framework == 'tflite':
        # convert_tflite_from_savedmodel(FLAGS.tflite)      #转换得到的tflite的input和output为fp32
        convert_tflite_from_keras(FLAGS.tflite)             #转换得到的tflite input和output为uint8
    elif FLAGS.framework == 'tf':
        model = getModelformCheckpoint(True)
        model.save(FLAGS.savePrefix)
    else:
        pass

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass