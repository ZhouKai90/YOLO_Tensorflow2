import glob, random
from absl import app, flags
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import core.utils as utils
import os
from converter.utils import get_graph
from tensorflow.python.saved_model import signature_constants

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
# if gpus:
#     tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
#     # tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

# target = 'helmet'
target = 'pedestrians'
# target = 'head'

# backbone = 'mobilenetv2'
# backbone = 'tiny-yolov3'
# backbone = 'yolov3'
backbone = 'peleenet'

flags.DEFINE_string('quantize_mode',                      'int8',            'fp32 or fp16 or int8 or None')

if target == 'helmet':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    if backbone == 'mobilenetv2':
        flags.DEFINE_string('trtPrefix', 'save/helmet/helmet_mobilenetv2_yolov3_540_960','path to save tflite')
    elif backbone == 'tiny-yolov3':
        flags.DEFINE_string('trtPrefix', 'save/helmet/helmet_tiny_yolov3_540_960_test', 'path to save tflite')

elif target == 'pedestrians':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    if backbone == 'mobilenetv2':
        flags.DEFINE_string('trtPrefix', 'save/pedestrian/pedestrians_mobilenetv2_yolov3_540_960',          'path to save tflite')
    elif backbone == 'tiny-yolov3':
        flags.DEFINE_string('trtPrefix', 'save/pedestrian/pedestrians_tiny_yolov3_540_960',          'path to save tflite')
    elif backbone == 'yolov3':
        flags.DEFINE_string('trtPrefix', 'save/pedestrian/pedestrians_yolov3_480_640',          'path to save tflite')
    elif backbone == 'peleenet':
        flags.DEFINE_string('savePrefix', 'save/pedestrian/savedmodel/peleenet_yolov3_540_960', 'path to savedmodel')
        flags.DEFINE_string('trtPrefix', 'save/pedestrian/tensorrt/pleenet_yolov3_540_960', 'path to save tflite')

elif target == 'head':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    flags.DEFINE_string('quan_images',  'data/quan/head',                     'images to quan')
    if backbone == 'peleenet':
        flags.DEFINE_string('trtPrefix', 'save/head/tensorrt/head_peleenet_yolov3_540_960', 'path to save tflite')
        flags.DEFINE_string('savePrefix', 'save/head/savedmodel/peleenet_yolov3_540_960', 'path to savedmodel')

# Define a generator function that yields input data, and run INT8
# calibration with the data. All input data should have the same shape.
# At the end of convert(), the calibration stats (e.g. range information)
# will be saved and can be used to generate more TRT engines with different
# shapes. Also, one TRT engine will be generated (with the same shape as
# the calibration data) for save later.
def representative_data_gen(imgCnt=0):
  imgDir = os.path.abspath(FLAGS.quan_images)
  imgNameList = glob.glob(os.path.join(imgDir, '*.jpg'))
  imgIndex = random.sample(range(len(imgNameList)), len(imgNameList) if imgCnt == 0 else imgCnt)
  for i in imgIndex:
    print(imgNameList[i])
    imgMat = cv2.imread(imgNameList[i])
    imgMat = utils.image_preprocess(np.copy(imgMat), [FLAGS.input_size[0], FLAGS.input_size[1]])
    imgMat = imgMat[np.newaxis, :, :, :]
    yield [tf.constant(imgMat.astype(np.float32))]

def convert_tf2_1():
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    # 分配给tensorRT的显存大小
    params._replace(max_workspace_size_bytes=4000000000)
    params._replace(max_batch_size=8)
    # params._replace(maximum_cached_engines=100)
    if FLAGS.quantize_mode == 'int8':
        params._replace(precision_mode=trt.TrtPrecisionMode.INT8)
        params._replace(use_calibration=True)
    elif FLAGS.quantize_mode == 'float16':
        params._replace(precision_mode=trt.TrtPrecisionMode.FP16)
    else:
        params._replace(precision_mode=trt.TrtPrecisionMode.FP32)
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=FLAGS.savePrefix,
        conversion_params=params)
    converter.convert()
    # converter.convert(calibration_input_fn=representative_data_gen if FLAGS.quantize_mode == 'int8' else None)
    if FLAGS.quantize_mode != 'fp32':
        converter.build(input_fn=representative_data_gen)
    converter.save(output_saved_model_dir=FLAGS.trtPrefix+'_{}'.format(FLAGS.quantize_mode))
    print('Done Converting to TF-TRT')

    saved_model_loaded = tf.saved_model.load(FLAGS.trtPrefix+'_{}'.format(FLAGS.quantize_mode))
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    trt_graph = graph_func.graph.as_graph_def()
    for n in trt_graph.node:
        print(n.op)
        if n.op == "TRTEngineOp":
            print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
        else:
            print("Exclude Node: %s, %s" % (n.op, n.name.replace("/", "_")))
    print("model saved to: {}".format(FLAGS.trtPrefix))

    trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
    print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
    all_nodes = len([1 for n in trt_graph.node])
    print("numb. of all_nodes in TensorRT graph:", all_nodes)

def convert_tf2_4():
  #without pre-build engines
    if FLAGS.quantize_mode == 'fp16':
        params = tf.experimental.tensorrt.ConversionParams(
          precision_mode='FP16')
        converter = tf.experimental.tensorrt.Converter(
          input_saved_model_dir=FLAGS.savePrefix, conversion_params=params)
        converter.convert()
    elif FLAGS.quantize_mode == 'fp32':
        params = tf.experimental.tensorrt.ConversionParams(
          precision_mode='FP32',
          # Set this to a large enough number so it can cache all the engines.
          maximum_cached_engines=16)
        converter = tf.experimental.tensorrt.Converter(
          input_saved_model_dir=FLAGS.savePrefix, conversion_params=params)
        converter.convert()

        # Define a generator function that yields input data, and use it to execute
        # the graph to build TRT engines.
        # With TensorRT 5.1, different engines will be built (and saved later) for
        # different input shapes to the TRTEngineOp.

        converter.build(input_fn=representative_data_gen)  # Generate corresponding TRT engines
    elif FLAGS.quantize_mode == 'int8':
        params = tf.experimental.tensorrt.ConversionParams(
          precision_mode='INT8',
          # Currently only one INT8 engine is supported in this mode.
          maximum_cached_engines=1,
          use_calibration=True)
        converter = tf.experimental.tensorrt.Converter(
          input_saved_model_dir=FLAGS.savePrefix, conversion_params=params)

        converter.convert(calibration_input_fn=representative_data_gen)

        # (Optional) Generate more TRT engines offline (same as the previous
        # option), to avoid the cost of generating them during inference.
        converter.build(input_fn=representative_data_gen)

    # Save the TRT engine and the engines.
    converter.save(FLAGS.trtPrefix)

def main(_argv):
    convert_tf2_1()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
