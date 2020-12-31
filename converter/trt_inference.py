import os
from time import time
from core.utils import image_preprocess, draw_bbox, postprocess_boxes, nms
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
    # tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
    # 对需要进行限制的GPU进行设置
    tf.config.experimental.set_virtual_device_configuration(gpus[1],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=2048)])

# detection_type = 'helmet'
detection_type = 'pedestrian'
# detection_type = 'head'

parser = argparse.ArgumentParser("Run TF-Lite YOLO-V3 Tiny inference.")

# stride = [16,32]
# stride = [8,16,32]
stride = [16, 32, 64]

if detection_type == 'helmet':
    parser.add_argument("--model", default='save/helmet/helmet_tiny_yolov3_540_960.tflite',
                        type=str, help="Model to load.")
    parser.add_argument("--anchors", default='data/anchors/helmet_540_960_6_anchors.txt', type=str,
                        help="Anchors file.")
    parser.add_argument("--classes", default='data/classes/helmet.names', type=str, help="Classes (.names) file.")
elif detection_type == 'pedestrian':
    parser.add_argument("--model", default='save/pedestrian/tensorrt/pleenet_yolov3_540_960_int8',
                        type=str, help="Model to load.")
    parser.add_argument("--anchors", default='data/anchors/person_crowdhuman_540_960_9_anchors.txt', type=str,
                        help="Anchors file.")
    parser.add_argument("--classes", default='data/classes/pedestrian.names', type=str, help="Classes (.names) file.")
elif detection_type == 'head':
    parser.add_argument("--model", default='save/head/tensorrt/head_peleenet_yolov3_540_960_int8',
                        type=str, help="Model to load.")
    parser.add_argument("--anchors", default='data/anchors/head_crowdhuman_540_960_9_anchors.txt',
                        type=str, help="Anchors file.")
    parser.add_argument("--classes", default='data/classes/head.names',
                        type=str, help="Classes (.names) file.")
else:
    print("detection type not support")
    pass

parser.add_argument("--BGR2RGB", default=False, type=bool, help="BGR2RGB.")

parser.add_argument("--score_threshold", help="Detection threshold.", type=float, default=0.5)
parser.add_argument("--nms_threshold", help="NMS threshold.", type=float, default=0.45)

parser.add_argument("--image", default='data/images/pedestrian/', type=str, help="Run inference on image.")
parser.add_argument("--deteciton_out", default='data/detection/test/', type=str, help="detection out to save.")
parser.add_argument("--rtsp_url", default='rtsp://admin:starblaze123@172.16.65.40:554/h264/1/main/av_stream',
                    type=str, help="rtsp url.")
args = parser.parse_args()

def image_inf():
    saved_model_loaded = tf.saved_model.load(args.model, tags=[tag_constants.SERVING])
    signature_keys = list(saved_model_loaded.signatures.keys())
    print(signature_keys)
    graph_func = saved_model_loaded.signatures['serving_default']
    # graph_func = saved_model_loaded.signatures[
    #     trt.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]  # 获取推理函数
    # frozen_func = trt.convert_to_constants.convert_variables_to_constants_v2(
    #     graph_func)  # 将模型中的变量变成常量,这一步可以省略,直接调用graph_func也行

    input_shape = [540, 960]
    imgNameList = os.listdir(args.image)
    for img in imgNameList:
        if os.path.isfile(img):
            continue
        print(img)

        imgMat = cv2.imread(args.image+img)
        # Crop frame to network input shape
        if args.BGR2RGB is True:
            imgData = cv2.cvtColor(np.copy(imgMat), cv2.COLOR_BGR2RGB)
        else:
            imgData = np.copy(imgMat)
        imgData = image_preprocess(imgData, input_shape)

        # Add batch dimension
        imgData = imgData[np.newaxis, ...].astype(np.float32)

        # print(f"image_preprocess speed: : {(time()-start) * 1000} ms.")

        # Run inference, get boxes
        timeNode1 = time()
        pred_bbox = []
        result = graph_func(tf.constant(imgData))
        timeNode2 = time()
        print("Net forward pass time: ", (timeNode2-timeNode1)*1000, ' ms')
        for key, value in result.items():
            print(key)
            pred_bbox.append(value)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = postprocess_boxes(pred_bbox, imgMat.shape[:2], input_shape, args.score_threshold)
        bboxes = nms(bboxes, args.nms_threshold, method='nms')

        if len(bboxes) > 0:
            print('bbox detected: ', len(bboxes))
            imgMat = draw_bbox(imgMat, bboxes, args.classes, False)
            cv2.imwrite(args.deteciton_out+img, imgMat)

if __name__ == "__main__":
    print(args.model)
    image_inf()
    # video_inf(interpreter, rtspURL)
