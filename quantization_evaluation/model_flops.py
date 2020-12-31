# coding=utf-8
import tensorflow as tf
from core.yolov3 import YOLOv3, tiny_YOLOv3
from core.model_factory import decode

input_size   = 416
tiny         = True

# 浮点运行次数
# FLOPS：注意全大写，是floating point operations per second的缩写，意指每秒浮点运算次数，理解为计算速度。是一个衡量硬件性能的指标。
# FLOPs：注意s小写，是floating point operations的缩写（s表复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。
# In TF 2.x you have to use tf.compat.v1.RunMetadata instead of tf.RunMetadata
# To work your code in TF 2.1.0, i have made all necessary changes that are compliant to TF 2.x

# 必须要下面这行代码
tf.compat.v1.disable_eager_execution()
print(tf.__version__)


# 我自己使用的函数
def get_flops_params():
    sess = tf.compat.v1.Session()
    graph = sess.graph
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph,
                                           options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph, run_meta=run_meta,
                                          cmd='op', options=opts)
    return flops.total_float_ops  # Prints the "flops" of the model.


if __name__ == '__main__':
    input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = tiny_YOLOv3(input_layer) if tiny else YOLOv3(input_layer)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)

    # 获取模型每一层的参数详情
    # model.summary()
    # 获取模型浮点运算总次数和模型的总参数
    get_flops_params()