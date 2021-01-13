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

target = 'head'

# backbone = 'yolov3'
backbone = 'peleenet'

flags.DEFINE_string('quantize_mode',                      'fp16',            'fp32 or fp16 or int8 or None')

if target == 'head':
    flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
    flags.DEFINE_string('quan_images',  'data/quan/head',                     'images to quan')
    flags.DEFINE_string('PBPath', 'save/head/PB/',          'path to save tflite')
    if backbone == 'peleenet':
        flags.DEFINE_string('PBPrefix', 'head_peleenet_yolov3_540_960.pb',          'path to save tflite')
        flags.DEFINE_string('H5File', 'save/head/savedmodel/peleenet_yolov3_540_960.h5', 'path to savedmodel')
    elif backbone == 'yolov3':
        flags.DEFINE_string('PBPrefix', 'head_yolov3_540_960.pb',          'path to save tflite')
        flags.DEFINE_string('H5File', 'save/head/savedmodel/yolov3_540_960.h5', 'path to savedmodel')

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        #         freeze_var_names = list(set(v.op.name for v in tf1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        #         output_names += [v.op.name for v in tf1.global_variables()]
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
        print("output_names:", output_names)
        return outgraph

def main(_argv):
    # 加载hdf5模型
    hdf5_pb_model = tf1.keras.models.load_model(FLAGS.H5File)
    frozen_graph = freeze_session(tf1.compat.v1.keras.backend.get_session(),
                                  output_names=[out.op.name for out in hdf5_pb_model.outputs])
    tf1.train.write_graph(frozen_graph, FLAGS.PBPath, FLAGS.PBPrefix, as_text=False)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass