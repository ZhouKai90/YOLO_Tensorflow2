import cv2
import os
from time import time
import numpy as np
import onnxruntime as ort
from core.utils import get_anchors, read_class_names, image_preprocess, draw_bbox, postprocess_boxes, nms
from converter.utils import inference_decode

class modelConfig(object):
    ONNX_MODEL = "save/head/ONNX/yolov3_540_960_deploy.onnx"
    # ONNX_MODEL = "save/head/ONNX/peleenet_yolov3_540_960_deploy.onnx"
    ANCHOR_FILE = "data/anchors/head_crowdhuman_540_960_9_anchors.txt"
    CLASSES_FILE = "data/classes/head.names"
    SCORE_THRESHOLD = 0.5
    NMS_THRESHOLD   = 0.45
    INPUT_NODE = 'input_1:0'
    INPUT_SHAPE = [3, 540, 960]
    STRIDE = [8, 16, 32]
    # STRIDE = [16, 32, 64]
    # we want the outputs in this order
    OUTPUT_NODE = ['Identity_2:0', 'Identity_1:0', 'Identity:0']
    INPUT_IMG_PATH = "data/images/testhead/"
    OUTPUT_IMG_PATH = "data/images/detection_out/test/"

def inference(sess, oriImgShape, imgData, anchors, stride, classNum):
    timeNode1 = time()
    # Run model
    inferenceResult = sess.run(modelConfig.OUTPUT_NODE, {modelConfig.INPUT_NODE: imgData})
    timeNode2 = time()
    print("Net forward-pass time:", (timeNode2-timeNode1)*1000, " ms")
    pred_bbox = []
    for n, tensor in enumerate(inferenceResult):
        out = inference_decode(tensor, anchors, stride, n, classNum)
        pred_bbox.append(np.reshape(out, (-1, np.shape(out)[-1])))

    pred_bbox = np.concatenate(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, oriImgShape, modelConfig.INPUT_SHAPE[1:3], modelConfig.SCORE_THRESHOLD)
    bboxes = nms(bboxes, modelConfig.NMS_THRESHOLD, method='nms')
    timeNode3 = time()
    # print(f"postprocess_boxes pass time: {(timeNode3-timeNode2)*1000} ms.")
    return bboxes

def main():
    anchors = get_anchors(modelConfig.ANCHOR_FILE)
    classes = read_class_names(modelConfig.CLASSES_FILE)

    sess = ort.InferenceSession(modelConfig.ONNX_MODEL)

    imgNameList = os.listdir(modelConfig.INPUT_IMG_PATH)
    for img in imgNameList:
        if os.path.isfile(img):
            continue
        print(img)

        start = time()
        imgMat = cv2.imread(modelConfig.INPUT_IMG_PATH+img)
        imgData = image_preprocess(np.copy(imgMat), modelConfig.INPUT_SHAPE[1:3], dtype='fp32')
        # Add batch dimension
        imgData = np.expand_dims(imgData.astype(np.float32), 0)
        # print(f"image_preprocess speed: : {(time()-start) * 1000} ms.")

        # Run inference, get boxes
        bboxes = inference(sess, imgMat.shape[:2], imgData, anchors, modelConfig.STRIDE, len(classes))
        if len(bboxes) > 0:
            print('bbox detected: ', len(bboxes))
            imgMat = draw_bbox(imgMat, bboxes, modelConfig.CLASSES_FILE, False)
            cv2.imwrite(modelConfig.OUTPUT_IMG_PATH+img, imgMat)

def check_onnx_model(onnx_model):
    import onnx
    model = onnx.load(onnx_model)
    onnx.checker.check_model(model)

if __name__ == '__main__':
    main()
