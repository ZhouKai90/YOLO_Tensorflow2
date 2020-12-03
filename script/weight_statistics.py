# coding: utf-8

import tensorflow.lite as tflite
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2

# modelFile = '../save/helmet/tiny_yolov3_helmet540_960_int8.tflite'
ped_savedMolde = '../save/pedestrian/savedmodel/mobilenetv2_yolov3_540_960'
hel_savedMolde = '../save/helmet/savedmodel/mobilenetv2_yolov3_540_960'

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[:2], device_type='GPU')
    for gpu in gpus[:2]:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

def savedModelAnalyze(writer, savedMolde, index):
    model = keras.models.load_model(savedMolde)
    layers = model.layers
    with writer.as_default():
        for layer in layers:
            if hasattr(layer, 'kernel'):
            # if 'conv2d' in layer.kernel.name:
            #     layerName = layer.name
                print(layer.name)
                tf.summary.histogram(layer.name, layer.weights[0], index)
    writer.flush()
    model.summary()
    # print(layer)
    pass

if __name__ == '__main__':
    writer = tf.summary.create_file_writer('./tensorboard_tflite')
    savedModelAnalyze(writer, ped_savedMolde, 1)
    savedModelAnalyze(writer, hel_savedMolde, 2)