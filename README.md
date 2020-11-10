# YOLO_Tensorflow2

1、基于tensorflow2.3，安全帽检测、行人检测和自定义数据集训练

2、支持yolov3、tiny-yolov3、mobilenetv2-yolov3模型

3、支持tflite转换和量化

4、sovedmodel、tflite推理测试demo



#### 注意

1、在转换tflite和量化过程中使用了tf2.1版本，并使用了tf.compat.v1.lite.TFLiteConverter.from_keras_model_file，以保证能和edgetpu compiler版本兼容。

#### TO DO

1、行人检测量化之后tflite推理异常



