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
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
    # tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
    # 对需要进行限制的GPU进行设置
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=2048)])

flags.DEFINE_string('savedmodel',   'save/head/savedmodel/peleenet_yolov3_540_960.h5',          'path to savedmodel')
flags.DEFINE_string('classes_file',        'data/classes/head.names',                               'classes file')

# flags.DEFINE_string('savedmodel',   'save/pedestrian/savedmodel/peleenet_yolov3_540_960.h5',          'path to savedmodel')
# flags.DEFINE_string('classes_file',        'data/classes/pedestrian.names',                               'classes file')

# flags.DEFINE_string('savedmodel',   'save/helmet/savedmodel/mobilenetv2_yolov3_540_960',          'path to savedmodel')
# flags.DEFINE_string('classes_file',        'data/classes/helmet.names',                               'classes file')

flags.DEFINE_string('detection_out_dir',  'data/images/detection_out/test/',                     'images to quan')
flags.DEFINE_string('images_dir',  'data/images/testhead/',                     'images to quan')
flags.DEFINE_multi_integer('input_size',  [540, 960],                       'define input size of export model')
flags.DEFINE_float('score_thres',   0.4,                                    'define score threshold')
flags.DEFINE_float('nms_thres',   0.45,                                    'define nms threshold')
flags.DEFINE_boolean('BGR2RGB',   False,                                    'convert BGR 2 RGB')


def main(_argv):
    model = keras.models.load_model(FLAGS.savedmodel)
    model.summary()
    # keras.utils.plot_model(model, 'model_info.png', show_shapes=True)

    imgDir = os.path.abspath(FLAGS.images_dir)
    imgNameList = glob.glob(os.path.join(imgDir, '*.jpg'))

    for img in imgNameList:
        print(img)
        name = img[img.rfind('/'):]
        original_image = cv2.imread(img)
        original_image_size = original_image.shape[:2]
        if FLAGS.BGR2RGB:
            image_data = cv2.cvtColor(np.copy(original_image), cv2.COLOR_BGR2RGB)
        else:
            image_data = np.copy(original_image)
        image_data = utils.image_preprocess(image_data, FLAGS.input_size)
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
            image = utils.draw_bbox(original_image, bboxes, classes_file=FLAGS.classes_file, show_label=False)
            cv2.imwrite(FLAGS.detection_out_dir+name, image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


