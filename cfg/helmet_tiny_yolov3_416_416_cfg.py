#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description :
#
#================================================================

from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

__C.YOLO.IS_TINY                = True
# Set the class name
__C.YOLO.CLASSES                = "data/classes/helmet.names"
__C.YOLO.ANCHORS                = "data/anchors/helmet_tiny_anchors.txt"
__C.YOLO.STRIDES                = [16, 32]          #for tiny yolo

__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5

# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "data/dataset/helmet/VOC2020/trainval.txt"
__C.TRAIN.BATCH_SIZE          = 20
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = [416]
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.DATA_TYPE           = 'uint8'  # float32 or uint8
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.EPOCHS              = 50
__C.TRAIN.CKPT_DIR            = 'save/ckpt/tiny_yolov3_input_uint8/'
__C.TRAIN.LOG_DIR             = 'save/log/tiny_yolov3_input_uint8/'
__C.TRAIN.PRETRAIN            = None
# __C.TRAIN.PRETRAIN            = 'save/ckpt/tiny_yolov3/model_epoch15'



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "data/dataset/helmet/VOC2020/test.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 544
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.3
__C.TEST.IOU_THRESHOLD        = 0.45


