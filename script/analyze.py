# coding: utf-8

import tensorflow.lite as tflite
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# modelFile = '../save/helmet/helmet_mobilenetv2_yolov3_540_960.tflite'
# IOQmodelFile = '../save/helmet/helmet_mobilenetv2_yolov3_540_960_IO_fp32.tflite'

# modelFile = '../save/pedestrian/pedestrians_tiny_yolov3_540_960.tflite'
# IOQmodelFile = '../save/pedestrian/pedestrians_tiny_yolov3_540_960_IO_fp32.tflite'

# modelFile = '../save/pedestrian/tflite/tiny_yolov3_pedstrian_540_960_1W_final.tflite'
# IOQmodelFile = '../save/pedestrian/tflite/tiny_yolov3_pedstrian_540_960_1W_int8_final.tflite'

# modelFile = '../save/helmet/helmet_tiny_yolov3_helmet540_960.tflite'
# IOQmodelFile = '../save/helmet/helmet_tiny_yolov3_helmet540_960_IO_fp32.tflite'


tiny_yolo_layerName = [
    'model/conv2d/Conv2D/ReadVariableOp/resource',
    'model/conv2d/Conv2D_bias',
    'model/max_pooling2d/MaxPool',
    'model/conv2d_1/Conv2D/ReadVariableOp/resource',
    'model/conv2d_1/Conv2D_bias',
    'model/max_pooling2d_1/MaxPool',
    'model/conv2d_2/Conv2D/ReadVariableOp/resource',
    'model/conv2d_2/Conv2D_bias',
    'model/max_pooling2d_2/MaxPool',
    'model/conv2d_3/Conv2D/ReadVariableOp/resource',
    'model/conv2d_3/Conv2D_bias',
    'model/max_pooling2d_3/MaxPool',
    'model/conv2d_4/Conv2D/ReadVariableOp/resource',
    'model/conv2d_4/Conv2D_bias',
    'model/max_pooling2d_4/MaxPool',
    'model/conv2d_5/Conv2D/ReadVariableOp/resource',
    'model/conv2d_5/Conv2D_bias',
    'model/max_pooling2d_5/MaxPool',
    'model/conv2d_6/Conv2D/ReadVariableOp/resource',
    'model/conv2d_6/Conv2D_bias',
    'model/conv2d_7/Conv2D/ReadVariableOp/resource',
    'model/conv2d_7/Conv2D_bias',
    'model/conv2d_8/Conv2D/ReadVariableOp/resource',
    'model/conv2d_8/Conv2D_bias',
    'model/conv2d_9/Conv2D/ReadVariableOp/resource',
    'model/conv2d_9/Conv2D_bias',
    'model/conv2d_10/Conv2D/ReadVariableOp/resource',
    'model/conv2d_10/Conv2D_bias',
    'model/tf_op_layer_resize/ResizeNearestNeighbor/resize/ResizeNearestNeighbor',
    # 'model/tf_op_layer_resize/ResizeBilinear/resize/ResizeBilinear',
    'model/tf_op_layer_resize/ResizeNearestNeighbor/resize/ResizeNearestNeighbor/size',
    # 'model/tf_op_layer_resize/ResizeBilinear/resize/ResizeBilinear/size',
    'model/tf_op_layer_concat/concat',
    # 'model/tf_op_layer_add/add',
    'model/conv2d_11/Conv2D/ReadVariableOp/resource',
    'model/conv2d_11/Conv2D_bias',
    'model/conv2d_12/Conv2D/ReadVariableOp/resource',
    'model/conv2d_12/Conv2D_bias',
    'Identity',
    'Identity_1'
]

mobilenet_layer = [
    'model/conv2d_5/Conv2D/ReadVariableOp/resource',
    'model/conv2d_5/Conv2D_bias',
    'model/tf_op_layer_resize/ResizeNearestNeighbor/resize/ResizeNearestNeighbor',
    'model/tf_op_layer_resize/ResizeNearestNeighbor/resize/ResizeNearestNeighbor/size',
    'model/conv2d_2/Conv2D/ReadVariableOp/resource',
    'model/conv2d_2/Conv2D_bias',
    'model/tf_op_layer_concat/concat',
    'model/conv2d_6/Conv2D/ReadVariableOp/resource',
    'model/conv2d_6/Conv2D_bias',
    'model/conv2d_7/Conv2D/ReadVariableOp/resource',
    'model/conv2d_7/Conv2D_bias',
    'Identity',
    'Identity_1'
]
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[:2], device_type='GPU')
    for gpu in gpus[:2]:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)


def inference(interpreter, imgMat):
    input_details = interpreter.get_input_details()
    # if args.input_output_tensor_quant:
    if input_details[0]['dtype'] == np.uint8:
        # 如果input和output都量化为INT8，就直接输入图片推理
        imgData = imgMat.astype(np.uint8)
    else:
        print("Input tensor not quant")
        # Normalize image from 0 to 1
        imgData = np.divide(imgMat, 255.).astype(np.float32)
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], imgData)
    # Run model
    interpreter.invoke()

def layerOutputCompile(interpreterTFLite, interpreterPTQ):
    allTensorFP32 = interpreterTFLite.get_tensor_details()
    allTensorINT8 = interpreterPTQ.get_tensor_details()

    for tensorFP32 in allTensorFP32:
        tensorINT8Name = None
        for tensorINT8 in allTensorINT8:
            if tensorFP32['name'] == tensorINT8['name']:
                tensorINT8Name = tensorINT8['name']
                break
        assert tensorINT8Name != None
        outFP32 = interpreterTFLite.get_tensor(tensorFP32['index'])
        outFP32 = outFP32.astype(np.float32)

        outINT8 = interpreterPTQ.get_tensor(tensorINT8['index'])
        if tensorINT8['quantization'] != (0, 0):
            print('quantization')
            scale, zero = tensorINT8['quantization']
            outINT8 = (outINT8.astype(np.float32) - zero) * scale
        # out = (out.astype(np.float32) - zero) * scale
        # d1 = mean_squared_error(outFP32, outINT8)
        d1 = cosine_distance(outFP32.flatten(), outINT8.flatten())

        print(tensorFP32['name'], ':', d1)


def oneLayerCompile(layerName, interpreterTFLite, interpreterPTQ):
    allTensorFP32 = interpreterTFLite.get_tensor_details()
    allTensorINT8 = interpreterPTQ.get_tensor_details()
    tensorFP32 = None
    tensorINT8 = None

    for tmpFP32 in allTensorFP32:
        if tmpFP32['name'] == layerName:
            tensorFP32 = tmpFP32
    for tmpINT8 in allTensorINT8:
        if tmpINT8['name'] == layerName:
            tensorINT8 = tmpINT8
    assert tensorFP32 != None and tensorINT8 != None

    outFP32 = interpreterTFLite.get_tensor(tensorFP32['index'])
    outFP32 = outFP32.astype(np.float32)

    outINT8 = interpreterPTQ.get_tensor(tensorINT8['index'])
    if tensorINT8['quantization'] != (0, 0):
        print('quantization')
        scale, zero = tensorINT8['quantization']
        outINT8 = (outINT8.astype(np.float32) - zero) * scale

    outFP32 = outFP32.flatten()
    outINT8 = outINT8.flatten()
    # np.save('./{}_FP32'.format(layerName), outFP32)
    # np.save('./{}_INT8'.format(layerName), outINT8)
    return cosine_distance(outFP32, outINT8)
    # return mean_squared_error(outFP32, outINT8)

def inputDataCompile(interpreterTFLite, interpreterPTQ):
    allTensorFP32 = interpreterTFLite.get_tensor_details()
    allTensorINT8 = interpreterPTQ.get_tensor_details()
    tensorFP32 = None
    tensorINT8 = None

    for tmpFP32 in allTensorFP32:
        if tmpFP32['name'] == 'input_1':
            tensorFP32 = tmpFP32
    for tmpINT8 in allTensorINT8:
        if tmpINT8['name'] == 'input_1_int8':
            tensorINT8 = tmpINT8
    assert tensorFP32 != None and tensorINT8 != None

    outFP32 = interpreterTFLite.get_tensor(tensorFP32['index'])
    outFP32 = outFP32.astype(np.float32)

    outINT8 = interpreterPTQ.get_tensor(tensorINT8['index'])
    scale, zero = tensorINT8['quantization']
    print(scale, ', ', zero)
    outINT8 = (outINT8.astype(np.float32) - zero) * scale
    return cosine_distance(outFP32.flatten(), outINT8.flatten())

def data_comp(interpreter):
    allTensorINT8 = interpreterPTQ.get_tensor_details()
    input = None
    input_int8 = None
    for tmpINT8 in allTensorINT8:
        if tmpINT8['name'] == 'Identity_1_int8':
            input_int8 = tmpINT8
        if tmpINT8['name'] == 'Identity_1':
            input = tmpINT8
    assert input != None and input_int8 != None

    outInputInt8 = interpreter.get_tensor(input_int8['index'])
    if input_int8['quantization'] != (0, 0):
        scale, zero = input_int8['quantization']
        outInputInt8 = (outInputInt8.astype(np.float32) - zero) * scale

    outInput = interpreter.get_tensor(input['index'])
    print('cosine_distance: ', cosine_distance(outInputInt8.flatten(), outInput.flatten()))

def cosine_distance(a, b):
    if a.shape != b.shape:
        return None
    # a = np.abs(a)
    # b = np.abs(b)
    num = float(np.matmul(a, b))
    s = np.linalg.norm(a) * np.linalg.norm(b)
    # print('a: ', np.linalg.norm(a))
    # print('b: ', np.linalg.norm(b))
    if s == 0:
        result = 0.0
    else:
        result = num / s
    return 1 - result


def featureCompile():
    featureINUT8 = np.load('./a.npy')
    featureFP32 = np.load('./b.npy')

    print(cosine_distance(featureINUT8, featureFP32))



if __name__ == '__main__':
    # featureCompile()
    # exit()
    imgMat = cv2.imread('./test.jpg')
    imgMat = np.expand_dims(imgMat, 0)
    interpreterTFLite = tflite.Interpreter(model_path=modelFile)
    interpreterTFLite.allocate_tensors()
    interpreterPTQ = tflite.Interpreter(model_path=IOQmodelFile)
    interpreterPTQ.allocate_tensors()
    inference(interpreterTFLite, np.copy(imgMat))
    inference(interpreterPTQ, np.copy(imgMat))
    # data_comp(interpreterPTQ)
    # print('iputdata: ', inputDataCompile(interpreterTFLite, interpreterPTQ))

    # layerName = ['model/tf_op_layer_resize/ResizeNearestNeighbor/resize/ResizeNearestNeighbor',
    #             'model/tf_op_layer_resize/ResizeNearestNeighbor/resize/ResizeNearestNeighbor/size',
    #             'model/tf_op_layer_concat/concat',]
    # layerName = ['Identity', 'Identity_1']
    # for name in tiny_yolo_layerName:
    #     d = oneLayerCompile(name, interpreterTFLite, interpreterPTQ)
    #     print(name, ':', d)

    layerOutputCompile(interpreterTFLite, interpreterPTQ)
    # tflite_analyze_tflite_ioq(imgMat)
    # tflite_analyze_tflite(imgMat, interpreter)
