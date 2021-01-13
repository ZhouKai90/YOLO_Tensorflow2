# YOLO_Tensorflow2

1、基于tensorflow2.3，安全帽检测、行人检测、人头检测和自定义数据集训练

2、支持yolov3、tiny-yolov3、mobilenetv2-yolov3、peleenet-yolov3模型

3、支持tflite量化和推理

4、支持savedmodel to ONNX转换和ONNX推理

5、支持ONNX to TensorRT模型推理

6、测试TF-TRT python接口加速推理



#### 注意

1、在转换tflite和量化过程中使用了tf2.1版本，并使用了tf.compat.v1.lite.TFLiteConverter.from_keras_model_file，以保证能和edgetpu compiler版本兼容。

2、尝试对行人检测进行QAT，但是因为tf model optimization并不支持上采样相关的层，所以导致失败

3、尝试过PB2uff，然后再用TensorRT推理，但是中途遇到一些问题，并且TRT-7之后不再跟新对uff和caffemodel的支持

4、测试过TF-TRT加速方式，确实对接口很友好，TRT支持加速的OP用TRT优化，不支持的OP直接用TF自己的实现，可以说是无缝部署。但是实际要C++上线需要源码编译TF并且支持TensorRT，bazel不熟悉，并且对边缘端实际不太友好，所以放弃。

5、savedmodel2ONNX之后，用onnxruntime的测试正确性，同时也有TensorRT Backend For ONNX的方案，但是想着既然都要用TRT了，就不再折腾用ONNX封装过的版本，不知道好不好用。

6、TensorRT测试过6.015_cu10.1版本，但是在解析onnx的时候出现了错误，看了git上的一些解决方案，遂将TRT升级到7.2.0.14，问题解决

#### TO DO

1、行人检测量化之后tflite推理异常

本工程是参考https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3，在此基础上进行的修改，特此说明

