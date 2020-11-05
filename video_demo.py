import tensorflow as tf
import os
import glob
import cv2
import numpy as np
import core.utils as utils
from tensorflow import keras
import time
from absl import flags, app
from absl.flags import FLAGS

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[1], enable=True)
    # for gpu in gpus[0:2]:

flags.DEFINE_string('savedmodel',   'save/pedestrian/savedmodel/mobilenetv2_yolov3_540_960',          'path to savedmodel')
flags.DEFINE_string('detection_out_dir',  'data/detection/540x960/mobilenetv2_yolov3_tf/',                     'images to quan')

flags.DEFINE_float('score_thres',   0.5,                                    'define score threshold')
flags.DEFINE_float('nms_thres',   0.45,                                    'define nms threshold')
flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')

flags.DEFINE_string('classes_file',        'data/classes/pedestrian.names',       'classes file')
flags.DEFINE_string('rtsp_url',  "rtsp://admin:starblaze123@172.16.65.40:554/h264/1/main/av_stream",
                   'define nms threshold')

def main(_argv):
    model = keras.models.load_model(FLAGS.savedmodel)
    model.summary()
    # keras.utils.plot_model(model, 'model_info.png', show_shapes=True)
    vid = cv2.VideoCapture(FLAGS.rtsp_url)
    img_cnt = 0
    output_cnt = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pass
        else:
            raise ValueError("No image!")

        img_cnt = 0 if img_cnt > 2000 else img_cnt+1
        if img_cnt % 20 != 0:
            continue

        if img_cnt > 2000:
            img_cnt = 0

        original_image_size = frame.shape[:2]

        image_data = utils.image_preporcess(np.copy(frame), [FLAGS.input_size[0], FLAGS.input_size[1]])
        image_data = image_data[np.newaxis, ...]

        t1 = time.time()
        pred_bbox = model.predict(image_data)
        preTime = time.time() - t1
        print("Prediction speed {} ms.".format(preTime * 1000))
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, FLAGS.input_size, FLAGS.score_thres)
        bboxes = utils.nms(bboxes, FLAGS.nms_thres, method='nms')
        if len(bboxes) > 0:
            print('Prediction box:', len(bboxes))
            image = utils.draw_bbox(frame, bboxes, classes_file=FLAGS.classes_file)
            output_cnt = 0 if output_cnt > 100 else output_cnt + 1
            cv2.imwrite(FLAGS.detection_out_dir+str(output_cnt)+'.jpg', image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


