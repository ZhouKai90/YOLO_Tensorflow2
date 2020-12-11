# coding=utf-8
import os
import tensorflow as tf
import cv2
import numpy as np
import core.utils as utils
from core.model_factory import get_model
from tqdm import tqdm
from cfg.config import CFG

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[2], device_type='GPU')

imageIDs = open(CFG.DATASET.IMAGESETS+'test.txt').read().strip().split()
inputPath = 'mAP/input'
DRPath = 'mAP/input/detection-results/'
DROutputPath = 'mAP/detection-img-output/'

CKPT   = CFG.TEST.CKPT
classes = utils.read_class_names(CFG.YOLO.CLASSES)
inputSize = CFG.TRAIN.INPUT_SIZE
scoreThreshold = CFG.TEST.SCORE_THRESHOLD
IOUThreshold = CFG.TEST.IOU_THRESHOLD

if not os.path.exists(inputPath):
    os.makedirs(inputPath)
if not os.path.exists(DRPath):
    os.makedirs(DRPath)
if not os.path.exists(DROutputPath):
    os.makedirs(DROutputPath)

def inference(model, imageID):
    imgPath = CFG.DATASET.JPEGIMAGE + imageID + '.jpg'
    imgMat      = cv2.imread(imgPath)
    cv2.imwrite(DROutputPath + imageID + '.jpg', imgMat)
    # imageData      = cv2.cvtColor(np.copy(imgMat), cv2.COLOR_BGR2RGB)
    imageData = utils.image_preprocess(np.copy(imgMat), inputSize)
    imageData = imageData[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(imageData)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, imgMat.shape[:2], inputSize, scoreThreshold)
    bboxes = utils.nms(bboxes, IOUThreshold, method='nms')
    # image = utils.draw_bbox(imgMat, bboxes, CFG.YOLO.CLASSES)
    # cv2.imwrite(DROutputPath + imageID + '.jpg', image)
    return bboxes

def get_detection_result():
    model = get_model(train=False)
    model.load_weights(CKPT)
    model.summary()
    # keras.utils.plot_model(model, "./model.png", show_shapes=True)
    for imageID in tqdm(imageIDs):
        f = open(DRPath + imageID + ".txt", "w")
        bboxes = inference(model, imageID)
        for bbox in bboxes:
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            className = classes[int(bbox[5])]
            f.write("%s %s %s %s %s %s\n" % (className, score, str(int(coor[0])), str(int(coor[1])), str(int(coor[2])),str(int(coor[3]))))
        f.close()
    return

if __name__ == '__main__':
    get_detection_result()