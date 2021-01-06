import os
from absl import app, flags
from absl.flags import FLAGS
import tensorflow.compat.v1 as tf1
tf1.reset_default_graph()
tf1.keras.backend.set_learning_phase(0)  # 调用模型前一定要执行该命令
tf1.disable_v2_behavior()  # 禁止tensorflow2.0的行为

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
# if gpus:
#     tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
#     # tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

# target = 'helmet'
# target = 'pedestrians'
target = 'head'

# backbone = 'mobilenetv2'
# backbone = 'tiny-yolov3'
backbone = 'yolov3'
# backbone = 'peleenet'

flags.DEFINE_string('quantize_mode',                      'fp16',            'fp32 or fp16 or int8 or None')

if target == 'helmet':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    if backbone == 'mobilenetv2':
        flags.DEFINE_string('trtPrefix', 'save/helmet/helmet_mobilenetv2_yolov3_540_960','path to save tflite')
    elif backbone == 'tiny-yolov3':
        flags.DEFINE_string('trtPrefix', 'save/helmet/helmet_tiny_yolov3_540_960_test', 'path to save tflite')

elif target == 'pedestrians':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    if backbone == 'mobilenetv2':
        flags.DEFINE_string('trtPrefix', 'save/pedestrian/pedestrians_mobilenetv2_yolov3_540_960',          'path to save tflite')
    elif backbone == 'tiny-yolov3':
        flags.DEFINE_string('trtPrefix', 'save/pedestrian/pedestrians_tiny_yolov3_540_960',          'path to save tflite')
    elif backbone == 'yolov3':
        flags.DEFINE_string('trtPrefix', 'save/pedestrian/pedestrians_yolov3_480_640',          'path to save tflite')
    elif backbone == 'peleenet':
        flags.DEFINE_string('savePrefix', 'save/pedestrian/savedmodel/peleenet_yolov3_540_960', 'path to savedmodel')
        flags.DEFINE_string('trtPrefix', 'save/pedestrian/tensorrt/pleenet_yolov3_540_960', 'path to save tflite')

elif target == 'head':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    flags.DEFINE_string('quan_images',  'data/quan/head',                     'images to quan')
    if backbone == 'peleenet':
        flags.DEFINE_string('trtPrefix', 'save/head/tensorrt/head_peleenet_yolov3_540_960', 'path to save tflite')
        flags.DEFINE_string('savePrefix', 'save/head/savedmodel/peleenet_yolov3_540_960', 'path to savedmodel')
    elif backbone == 'yolov3':
        flags.DEFINE_string('PBPath', 'save/head/savedmodel/',          'path to save tflite')
        flags.DEFINE_string('PBPrefix', 'head_yolov3_540_960.pb',          'path to save tflite')
        flags.DEFINE_string('savePrefix', 'save/head/savedmodel/yolov3_540_960.h5', 'path to savedmodel')

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        #         freeze_var_names = list(set(v.op.name for v in tf1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        #         output_names += [v.op.name for v in tf1.global_variables()]
        print("output_names", output_names)
        input_graph_def = graph.as_graph_def()
        #         for node in input_graph_def.node:
        #             print('node:', node.name)
        print("len node1", len(input_graph_def.node))
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                                     output_names)

        outgraph = tf1.graph_util.remove_training_nodes(frozen_graph)  # 去掉与推理无关的内容
        print("##################################################################")
        for node in outgraph.node:
            print('node:', node.name)
        print("len node1", len(outgraph.node))
        return outgraph

def main(_argv):
    # 加载hdf5模型
    hdf5_pb_model = tf1.keras.models.load_model(FLAGS.savePrefix)
    frozen_graph = freeze_session(tf1.compat.v1.keras.backend.get_session(),
                                  output_names=[out.op.name for out in hdf5_pb_model.outputs])
    tf1.train.write_graph(frozen_graph, FLAGS.PBPath, FLAGS.PBPrefix, as_text=False)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass