import os, cv2
import numpy as np
from time import time
import tensorrt as trt
from trt_common import *
from core.utils import get_anchors, read_class_names, image_preprocess, draw_bbox, postprocess_boxes, nms
from converter.utils import inference_decode

TRT_LOGGER = trt.Logger()

class modelConfig(object):
    # ONNX_MODEL = "save/head/ONNX/yolov3_540_960_deploy.onnx"
    # ENGINE_PATH = "save/head/TRT-engine/yolov3_540_960_deploy.trt"
    # STRIDE = [8, 16, 32]
    # OUTPUT_SHAPES = [(1, 68, 120, 18), (1, 34, 60, 18), (1, 17, 30, 18)]
    ONNX_MODEL = "save/head/ONNX/peleenet_yolov3_540_960_deploy.onnx"
    ENGINE_PATH = "save/head/TRT-engine/peleenet_yolov3_540_960_deploy.trt"
    STRIDE = [16, 32, 64]
    OUTPUT_SHAPES = [(1, 34, 60, 18), (1, 9, 15, 18), (1, 17, 30, 18)]    #为啥这个输出是乱序的？？
    ANCHOR_FILE = "data/anchors/head_crowdhuman_540_960_9_anchors.txt"
    CLASSES_FILE = "data/classes/head.names"
    SCORE_THRESHOLD = 0.4
    NMS_THRESHOLD   = 0.45
    INPUT_NODE = 'input_1:0'
    INPUT_SHAPE = [1, 540, 960, 3]
    # we want the outputs in this order
    OUTPUT_NODE = ['Identity_2:0', 'Identity_1:0', 'Identity:0']
    INPUT_IMG_PATH = "data/images/head/"
    OUTPUT_IMG_PATH = "data/images/detection_out/test/"

    MAX_BATCH_SIZE = 1
    MAX_WORKSPACE = GiB(1)  #1G


def build_engine_onnx(onnx_file_path, engine_file_path, precision='fp32', max_batch_size=1, cache_file=None):
    """Builds a new TensorRT engine and saves it, if no engine presents"""
    # if os.path.exists(engine_file_path):
    #     print('{} TensorRT engine already exists. Skip building engine...'.format(precision))
    #     return
    print('Building {} TensorRT engine from onnx file...'.format(precision))
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as b, b.create_network(explicit_batch) as n, trt.OnnxParser(n, TRT_LOGGER) as p:
    # with trt.Builder(TRT_LOGGER) as b, b.create_network() as n, trt.OnnxParser(n, TRT_LOGGER) as p:
        b.max_workspace_size = modelConfig.MAX_WORKSPACE  # 1GB
        b.max_batch_size = modelConfig.MAX_BATCH_SIZE
        if precision == 'fp16':
            b.fp16_mode = True
        elif precision == 'int8':
            # from ..calibrator import Calibrator
            # b.int8_mode = True
            # b.int8_calibrator = Calibrator(cache_file=cache_file)
            pass
        elif precision == 'fp32':
            pass
        else:
            print('Engine precision not supported: {}'.format(precision))
            raise NotImplementedError
        # Parse model file
        print('Beginning ONNX file parsing')
        with open(onnx_file_path, 'rb') as model:
            p.parse(model.read())

        if p.num_errors:
            print('Parsing onnx file found {} errors.'.format(p.num_errors))
            for error in range(p.num_errors):
                print(p.get_error(error))
            exit()
        print('Completed parsing of ONNX file')
        # last_layer = n.get_layer(n.num_layers - 1)
        # n.mark_output(last_layer.get_output(0))
        n.get_input(0).shape = modelConfig.INPUT_SHAPE
        engine = b.build_cuda_engine(n)
        print(engine_file_path)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())

def get_engine(engine_file_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def main():
    anchors = get_anchors(modelConfig.ANCHOR_FILE)
    classes = read_class_names(modelConfig.CLASSES_FILE)
    imgNameList = os.listdir(modelConfig.INPUT_IMG_PATH)
    # Build a TensorRT engine.
    with get_engine(modelConfig.ENGINE_PATH) as engine, \
            engine.create_execution_context() as context:
        ''' 分配host，device端的buffer'''
        inputs, outputs, bindings, stream = allocate_buffers(engine)
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
            # bboxes = inference(engine, imgMat.shape[:2], imgData, anchors, modelConfig.STRIDE, len(classes))

            '''进行inference'''
            inputs[0].host = imgData
            timeNode1 = time()
            # Run model
            trtOutputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            timeNode2 = time()
            print("Net forward-pass time:", (timeNode2 - timeNode1) * 1000, " ms")
            trtOutputs = [output.reshape(shape) for output, shape in zip(trtOutputs, modelConfig.OUTPUT_SHAPES)]
            pred_bbox = []
            for n, tensor in enumerate(trtOutputs):
                out = inference_decode(tensor, anchors, modelConfig.STRIDE, n, len(classes))
                pred_bbox.append(np.reshape(out, (-1, np.shape(out)[-1])))

            pred_bbox = np.concatenate(pred_bbox, axis=0)

            bboxes = postprocess_boxes(pred_bbox, imgMat.shape[:2], modelConfig.INPUT_SHAPE[1:3],
                                       modelConfig.SCORE_THRESHOLD)
            bboxes = nms(bboxes, modelConfig.NMS_THRESHOLD, method='nms')

            if len(bboxes) > 0:
                print('bbox detected: ', len(bboxes))
                imgMat = draw_bbox(imgMat, bboxes, modelConfig.CLASSES_FILE, False)
                cv2.imwrite(modelConfig.OUTPUT_IMG_PATH+img, imgMat)

if __name__ == '__main__':
    main()
    # build_engine_onnx(modelConfig.ONNX_MODEL, modelConfig.ENGINE_PATH)