#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import cv2
import numpy as np
import core.utils as utils
from core.model_factory import decode
from core.yolov3 import YOLOv3, tiny_YOLOv3, mobilenetV2_YOLOv3


# input_size   = [416, 416]
input_size   = [540, 960]
image_path   = "data/images/pedestrian/16.jpg"
CKPT         = 'save/pedestrian/ckpt/mobilenetv2_yolov3_540_960/model_epoch28'
tiny         = True
classes_file = "data/classes/pedestrian.names"
# classes_file = "data/classes/helmet.names"

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')

original_image      = cv2.imread(image_path)
original_image_size = original_image.shape[:2]

# image_data      = cv2.cvtColor(np.copy(original_image), cv2.COLOR_BGR2RGB)
image_data = utils.image_preporcess(original_image, input_size)
image_data = image_data[np.newaxis, ...].astype(np.float32)

input_layer  = tf.keras.layers.Input([input_size[0], input_size[1], 3])
feature_maps = mobilenetV2_YOLOv3(input_layer)
# feature_maps = tiny_YOLOv3(input_layer) if tiny else YOLOv3(input_layer)
bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)
model = tf.keras.Model(input_layer, bbox_tensors)

model.load_weights(CKPT)
model.summary()

pred_bbox = model.predict(image_data)
pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
pred_bbox = tf.concat(pred_bbox, axis=0)

bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')

image = utils.draw_bbox(original_image, bboxes, classes_file)
cv2.imwrite('data/detection_3.jpg', image)


