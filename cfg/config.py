#! /usr/bin/env python
# coding=utf-8

from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

CFG                           = __C

# YOLO options
__C.YOLO                      = edict()
# Set the class name
__C.YOLO.CLASSES                = "data/classes/head.names"
__C.YOLO.ANCHORS                = "data/anchors/head_crowdhuman_540_960_9_anchors.txt"
# __C.YOLO.STRIDES                = [16, 32, 64]          #for tiny yolo
__C.YOLO.STRIDES                = [8, 16, 32]          #for tiny yolo
__C.YOLO.BRANCH_SIZE             = len(__C.YOLO.STRIDES)
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5

__C.DATASET                   = edict()
__C.DATASET.PATH              = "data/dataset/crowd_human/"
__C.DATASET.JPEGIMAGE         = __C.DATASET.PATH + 'trainval/'
# __C.DATASET.IMAGESETS         = __C.DATASET.PATH + 'ImageSets/Main/'
# __C.DATASET.ANNOTATIONS       = __C.DATASET.PATH + 'Annotations/'
__C.DATASET.TRAIN_ANNOT       = __C.DATASET.PATH + "train.txt"
__C.DATASET.TEST_ANNOT         = __C.DATASET.PATH + "val.txt"
# __C.DATASET.TEST_ANNOT        = __C.DATASET.PATH + "test.txt"

# Train options
__C.TRAIN                     = edict()
# __C.TRAIN.BACKBONE            = 'peleenet_yolov3'			# or mobilenetv2
__C.TRAIN.BACKBONE            = 'yolov3'			# or mobilenetv2
__C.TRAIN.BATCH_SIZE          = 6
__C.TRAIN.INPUT_SIZE          = [540, 960]  #[H, W]
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.BGR2RGB             = False
__C.TRAIN.DATA_TYPE           = 'float32'  # float32 or uint8
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.EPOCHS              = 50
__C.TRAIN.CKPT_DIR            = 'save/head/ckpt/yolov3_540_960/'
__C.TRAIN.LOG_DIR             = 'save/head/tensorboard/yolov3_540_960/'
__C.TRAIN.CONTINUE_EPOCH      = 0
__C.TRAIN.PRETRAIN            = None
# __C.TRAIN.PRETRAIN            = 'save/pedestrian/ckpt/tiny_yolov3/model_fi'

# TEST options
__C.TEST                      = edict()
__C.TEST.BATCH_SIZE           = 6
__C.TEST.INPUT_SIZE           = [540, 960]  #[H, W]
__C.TEST.CKPT                 = 'save/head/ckpt/yolov3_540_960/model_final'
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.3
__C.TEST.IOU_THRESHOLD        = 0.45


