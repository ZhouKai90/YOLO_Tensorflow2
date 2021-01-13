import cv2
import uff
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
# from tensorrt.parsers import uffparser
from tensorrt.legacy.parsers import uffparser

class ModelData(object):
    PLAN_MODEL = "save/head/PB/head_peleenet_yolov3_540_960_deploy.pb.plan"
    PB_MODEL    = 'save/head/PB/head_peleenet_yolov3_540_960_deploy.pb'
    INPUT_NODE = 'input_1'
    INPUT_SHAPE = (3, 540, 960)
    DTYPE = trt.legacy.infer.DataType.FLOAT
    OUTPUT_NODE = ['s_obj_conv2d/BiasAdd', 'm_obj_conv2d/BiasAdd', 'l_obj_conv2d/BiasAdd']
    MAX_BATCH_SIZE = 1
    MAX_WORKSPACE = 2 << 30  #2G

def main():
    G_LOGGER = trt.legacy.infer.ConsoleLogger(trt.legacy.infer.LogSeverity.INFO)

    # 将PB转换为UFF文件
    uff_model = uff.from_tensorflow_frozen_model(ModelData.PB_MODEL, ModelData.OUTPUT_NODE)
    parser = uffparser.create_uff_parser()
    parser.register_input(ModelData.INPUT_NODE, ModelData.INPUT_SHAPE, 0)
    parser.register_output(ModelData.OUTPUT_NODE)
    # 创建推理engine
    engine = trt.legacy.utils.uff_to_trt_engine(
        G_LOGGER,
        uff_model,
        parser,
        ModelData.MAX_BATCH_SIZE,
        ModelData.MAX_WORKSPACE,
        datatype=ModelData.DTYPE)

    trt.legacy.utils.cwrite_engine_to_file(ModelData.PLAN_MODEL, engine.serialize())

    engine = trt.legacy.utils.load_engine(G_LOGGER, ModelData.PLAN_MODEL)
    #生成engine和context
    context = engine.create_execution_context()
    engine = context.get_engine()
    #确认输入和输出的tensor数量
    print(engine.get_nb_bindings())
    assert (engine.get_nb_bindings() == 1+3)

    img = cv2.imread('data/iamges/head/1.jpg')
    img = img.astype(np.float32)

    # create output array to receive data
    OUTPUT_SIZE = 10
    output = np.zeros(OUTPUT_SIZE, dtype=np.float32)

    # 使用PyCUDA申请GPU显存并在引擎中注册
    # 申请的大小是整个batchsize大小的输入以及期望的输出指针大小。
    d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
    d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)

    # 引擎需要绑定GPU显存的指针。PyCUDA通过分配成ints实现内存申请。
    bindings = [int(d_input), int(d_output)]

    # 建立数据流
    stream = cuda.Stream()
    # 将输入传给cuda
    cuda.memcpy_htod_async(d_input, img, stream)
    # 执行前向推理计算
    context.enqueue(1, bindings, stream.handle, None)
    # 将预测结果传回
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # 同步
    stream.synchronize()

    #post process, 推理结果保存在output中


if __name__ == '__main__':
    main()