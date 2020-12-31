import numpy as np
from core.yolov3 import YOLOv3, tiny_YOLOv3, mobilenetV2_YOLOv3, peleenet_YOLOV3
from core.yolov4 import YOLOv4, tiny_YOLOv4
from core.model_factory import decode
from core.utils import read_class_names, get_anchors

def sigmoid_1(inx):
    if inx.all()>=0:
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))

def sigmoid(x):
    return .5 * (1 + np.tanh(.5 * x))

def sigmoid_2(x):
    return 1. / (1 + np.exp(-x))


def decode_modify(conv_output, ANCHORS, STRIDES, i=0, classes_num=2):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """

    conv_shape       = np.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = np.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + classes_num))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    y = np.tile(np.arange(output_size, dtype=np.int32)[:, np.newaxis], [1, output_size])
    x = np.tile(np.arange(output_size, dtype=np.int32)[np.newaxis, :], [output_size, 1])

    xy_grid = np.concatenate([x[:, :, np.newaxis], y[:, :, np.newaxis]], axis=-1)
    xy_grid = np.tile(xy_grid[np.newaxis, :, :, np.newaxis, :], [batch_size, 1, 1, 3, 1])
    # xy_grid = np.cast(xy_grid, np.float32)

    pred_xy = (sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (np.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = np.concatenate([pred_xy, pred_wh], axis=-1)

    pred_conf = sigmoid(conv_raw_conf)
    pred_prob = sigmoid(conv_raw_prob)

    return np.concatenate([pred_xywh, pred_conf, pred_prob], axis=-1)

def inference_decode(conv_output, ANCHORS, STRIDES, i=0, classes_num=2):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """

    conv_shape       = np.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1:3]

    conv_output = np.reshape(conv_output, (batch_size, output_size[0], output_size[1], 3, 5 + classes_num))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    xy_grid = np.meshgrid(np.arange(output_size[1]), np.arange(output_size[0]))
    xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    # xy_grid = np.cast(xy_grid, np.float32)

    pred_xy = (sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (np.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = np.concatenate([pred_xy, pred_wh], axis=-1)

    pred_conf = sigmoid(conv_raw_conf)
    pred_prob = sigmoid(conv_raw_prob)

    return np.concatenate([pred_xywh, pred_conf, pred_prob], axis=-1)

def get_graph(input_tensor, classes_file, anchors_file, backbone, withDecode=False):
    numClasses = len(read_class_names(classes_file))
    anchors = get_anchors(anchors_file)
    if backbone == 'yolov3':
        feature_maps = YOLOv3(input_tensor, numClasses)
    elif backbone == 'tiny-yolov3':
        feature_maps = tiny_YOLOv3(input_tensor, numClasses)
    elif backbone == 'mobilenetv2':
        feature_maps = mobilenetV2_YOLOv3(input_tensor, numClasses)
    elif backbone == 'tiny-yolov4':
        feature_maps = tiny_YOLOv4(input_tensor, numClasses)
    elif backbone == 'peleenet':
        feature_maps = peleenet_YOLOV3(input_tensor, numClasses)
    else:
        raise ValueError('backbone unknow.')

    bboxTensors = []
    if withDecode:
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode(fm, i, anchors, numClasses)
            bboxTensors.append(bbox_tensor)
    else:
        bboxTensors = feature_maps

    return bboxTensors

def representative_dataset_gen(input_size):
    for _ in range(20):
        print('range')
        yield [np.random.uniform(0.0, 1.0, size=(1, input_size[0], input_size[1], 3)).astype(np.float32)]
