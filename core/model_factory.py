import numpy as np
import tensorflow as tf
from core.yolov3 import YOLOv3, tiny_YOLOv3, mobilenetV2_YOLOv3
from core.yolov4 import YOLOv4, tiny_YOLOv4
from cfg.config import CFG
import core.utils as utils
from tensorflow.keras import backend as K

NUM_CLASS       = len(utils.read_class_names(CFG.YOLO.CLASSES))
STRIDES         = np.array(CFG.YOLO.STRIDES)
IOU_LOSS_THRESH = CFG.YOLO.IOU_LOSS_THRESH
anchors= utils.get_anchors(CFG.YOLO.ANCHORS, CFG.YOLO.BRANCH_SIZE)

def get_model(train=True):
    input_tensor = tf.keras.layers.Input([CFG.TRAIN.INPUT_SIZE[0], CFG.TRAIN.INPUT_SIZE[1], 3])
    if CFG.TRAIN.BACKBONE == 'yolov3':
        conv_tensors = YOLOv3(input_tensor, NUM_CLASS)
    elif CFG.TRAIN.BACKBONE == 'yolov4':
        conv_tensors = YOLOv4(input_tensor, NUM_CLASS)
    elif CFG.TRAIN.BACKBONE == 'tiny_yolov3':
        conv_tensors = tiny_YOLOv3(input_tensor, NUM_CLASS)
    elif CFG.TRAIN.BACKBONE == 'tiny_yolov4':
        conv_tensors = tiny_YOLOv4(input_tensor, NUM_CLASS)
    elif CFG.TRAIN.BACKBONE == 'mobilenetv2_yolov3':
        conv_tensors = mobilenetV2_YOLOv3(input_tensor, NUM_CLASS)
    else:
        raise ValueError('backbone unknow.')

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, i, anchors, NUM_CLASS)
        if train:
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    model = tf.keras.Model(input_tensor, output_tensors)
    return model

def decode(conv_output, i, ANCHORS, NUM_CLASS):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1:3]

    conv_output = tf.reshape(conv_output, (batch_size, output_size[0], output_size[1], 3, 5 + NUM_CLASS))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    # y = tf.tile(tf.range(output_size[0], dtype=tf.int32)[:, tf.newaxis], [1, output_size[0]])
    # x = tf.tile(tf.range(output_size[1], dtype=tf.int32)[tf.newaxis, :], [output_size[1], 1])
    # xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    # xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])

    # xy_grid = tf.meshgrid(tf.range(output_size[1]), tf.range(output_size[0]))
    # xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    # xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
    # xy_grid_1 = tf.cast(xy_grid, tf.float32)

    grid_shape = output_size  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    xy_grid = K.concatenate([grid_x, grid_y])
    xy_grid = K.cast(xy_grid, K.dtype(conv_output))

    # assert xy_grid_1 == xy_grid

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
