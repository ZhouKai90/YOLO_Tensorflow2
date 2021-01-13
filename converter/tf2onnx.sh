#!/bin/bash
saved_model_path=save/head/savedmodel/peleenet_yolov3_540_960_deploy
onnx_model_file=save/head/ONNX/peleenet_yolov3_540_960_deploy.onnx
input_name=input_1:0
output_name=Identity_2:0,Identity_1:0,Identity:0
opset=13
#python -m tf2onnx.convert --saved-model $saved_model_path  --opset $opset --output $onnx_model_file --inputs $input_name --outputs $output_name
#python -m tf2onnx.convert --saved-model $saved_model_path  --opset $opset --output $onnx_model_file

python -m tf2onnx.convert --saved-model save/head/savedmodel/peleenet_yolov3_540_960_deploy --opset 10 --output save/head/ONNX/peleenet_yolov3_540_960_deploy.onnx