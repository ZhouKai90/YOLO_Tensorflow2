import tensorflow.lite as tflite
import os
from time import time
from utils import *
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
EDGETPU_SHARED_LIB = "libedgetpu.so.1"

# detection_type = 'helmet'
detection_type = 'pedestrian'
# detection_type = 'snow_panther'

parser = argparse.ArgumentParser("Run TF-Lite YOLO-V3 Tiny inference.")
parser.add_argument("--branch", default=2, type=int, help="branch size")

if detection_type == 'helmet':
    parser.add_argument("--model", default='save/helmet/helmet_tiny_yolov3_helmet540_960.tflite',
                        type=str, help="Model to load.")
    parser.add_argument("--anchors", default='data/anchors/helmet_540_960_6_anchors.txt', type=str,
                        help="Anchors file.")
    parser.add_argument("--classes", default='data/classes/helmet.names', type=str, help="Classes (.names) file.")
elif detection_type == 'pedestrian':
    parser.add_argument("--model", default='save/pedestrian/pedestrians_tiny_yolov4_540_960_IO_uint8.tflite',
                        type=str, help="Model to load.")
    parser.add_argument("--anchors", default='data/anchors/pedestrian_540_960_6_anchors.txt', type=str,
                        help="Anchors file.")
    parser.add_argument("--classes", default='data/classes/pedestrian.names', type=str, help="Classes (.names) file.")
elif detection_type == 'snow_panther':
    parser.add_argument("--model", default='save/snow_panther/snow_panther_tiny_yolov3_540_960_IO_uint8.tflite',
                        type=str, help="Model to load.")
    parser.add_argument("--anchors", default='data/anchors/snow_panther_540_960_6_anchors.txt',
                        type=str, help="Anchors file.")
    parser.add_argument("--classes", default='data/classes/snow_panther.names',
                        type=str, help="Classes (.names) file.")
else:
    print("detection type not support")
    pass

parser.add_argument("--big_first", default=True, type=bool, help="big_first.")
parser.add_argument("--BGR2RGB", default=False, type=bool, help="BGR2RGB.")

parser.add_argument("--score_threshold", help="Detection threshold.", type=float, default=0.6)
parser.add_argument("--nms_threshold", help="NMS threshold.", type=float, default=0.45)

parser.add_argument("--image", default='data/images/pedestrian/', type=str, help="Run inference on image.")
parser.add_argument("--deteciton_out", default='data/detection/test/', type=str, help="detection out to save.")
parser.add_argument("--rtsp_url", default='rtsp://admin:starblaze123@172.16.65.40:554/h264/1/main/av_stream',
                    type=str, help="rtsp url.")
args = parser.parse_args()

def make_interpreter(model_file, edge_tpu=False):
    if edge_tpu:
        model_file, *device = model_file.split('@')
        return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB,
                                    {'device': device[0]} if device else {})
            ])
    else:
        return tflite.Interpreter(model_path=model_file)

def get_interpreter_details(interpreter):
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]
    return input_details, output_details, input_shape

# Run YOLO inference on the image, returns detected boxes
def inference(interpreter, oriImgShape, imgData, anchors, stride, classNum):
    input_details, output_details, net_input_shape = \
            get_interpreter_details(interpreter)
    if input_details[0]['dtype'] == np.uint8:
        #如果input和output都量化为UINT8，就直接输入图片推理
        imgData = imgData.astype(np.uint8)
    else:
        print("Input tensor not quant")
        # Normalize image from 0 to 1
        imgData = np.divide(imgData, 255.).astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], imgData)

    timeNode1 = time()
    # Run model
    interpreter.invoke()
    timeNode2 = time()
    # print("Net forward-pass time: {(timeNode2-timeNode1)*1000} ms.")

    pred_bbox = []
    # print(stride)
    for i in range(args.branch):
        if args.big_first:
            out = interpreter.get_tensor(output_details[1-i]['index'])
        else:
            out = interpreter.get_tensor(output_details[i]['index'])
        if output_details[i]['dtype'] == np.uint8 and output_details[i]['quantization'] != (0, 0):
            scale, zero = output_details[i]['quantization']
            out = (out.astype(np.float32) - zero) * scale
        out = decode(out, anchors, stride, i, classNum)
        pred_bbox.append(np.reshape(out, (-1, np.shape(out)[-1])))

    pred_bbox = np.concatenate(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, oriImgShape, net_input_shape, args.score_threshold)
    bboxes = nms(bboxes, args.nms_threshold, method='nms')
    timeNode3 = time()
    # print(f"postprocess_boxes pass time: {(timeNode3-timeNode2)*1000} ms.")

    return bboxes

def image_inf(interpreter):
    anchors = get_anchors(args.anchors, args.branch)
    classes = read_class_names(args.classes)
    input_details, output_details, input_shape = get_interpreter_details(interpreter)
    print('input_shape:', input_shape[1:3])

    imgNameList = os.listdir(args.image)
    for img in imgNameList:
        if os.path.isfile(img):
            continue
        print(img)

        start = time()
        imgMat = cv2.imread(args.image+img)
        # Crop frame to network input shape
        if args.BGR2RGB is True:
            imgData = cv2.cvtColor(np.copy(imgMat), cv2.COLOR_BGR2RGB)
        else:
            imgData = np.copy(imgMat)
        imgData = image_preprocess(imgData, input_shape[1:3])

        # Add batch dimension
        imgData = np.expand_dims(imgData, 0)
        # print(f"image_preprocess speed: : {(time()-start) * 1000} ms.")

        # Run inference, get boxes
        stride = [8,16,32]
        bboxes = inference(interpreter, imgMat.shape[:2], imgData, anchors, stride[3-args.branch :], len(classes))
        if len(bboxes) > 0:
            print('bbox detected: ', len(bboxes))
            imgMat = draw_bbox(imgMat, bboxes, classes)
            cv2.imwrite(args.deteciton_out+img, imgMat)

def video_inf(interpreter, path):
    cap = cv2.VideoCapture(path)

    anchors = get_anchors(args.anchors, args.branch)
    classes = read_class_names(args.classes)
    input_details, output_details, input_shape = get_interpreter_details(interpreter)

    imgCnt = 0
    outputCnt = 0
    # Load and process image
    while True:
        # Read frame from webcam
        # start = time()
        ret, frame = cap.read()
        if ret is not True:
            continue

        if imgCnt > 2000:
            imgCnt = 0
        imgCnt += 1

        if imgCnt % 20 != 0:
            continue

        if args.BGR2RGB:
            imgData = cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2RGB)
        else:
            imgData = np.copy(frame)
        imgData = image_preprocess(imgData, input_shape[1:3])

        # Add batch dimension
        imgData = np.expand_dims(imgData, 0)
        # Run inference, get boxes
        # print(f"image_preprocess speed: : {(time() - start) * 1000} ms.")

        stride = [8,16,32]
        bboxes = inference(interpreter, frame.shape[:2], imgData, anchors, stride[3-args.branch], len(classes))
        if len(bboxes) > 0:
            print('bboxes:', len(bboxes))
            outputCnt = 0 if outputCnt > 100 else outputCnt + 1
            image = draw_bbox(frame, bboxes, classes)
            cv2.imwrite(args.deteciton_out + 'image_{}.jpg'.format(outputCnt), image)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    # writer.release()

if __name__ == "__main__":
    print(args.model)
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    image_inf(interpreter)
    # video_inf(interpreter, rtspURL)
