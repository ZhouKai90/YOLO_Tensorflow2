#! /usr/bin/env python
# coding=utf-8

import os
import shutil
import numpy as np
import tensorflow as tf
from core.dataset import Dataset
from cfg.config import cfg
from core.model_factory import get_model, compute_loss
from tensorflow import keras

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[2:], device_type='GPU')
    for gpu in gpus[2:]:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

trainset = Dataset('train')
steps_per_epoch = len(trainset)
global_steps = tf.Variable(steps_per_epoch*cfg.TRAIN.CONTINUE_EPOCH+1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

optimizer = tf.keras.optimizers.Adam()

if os.path.exists(cfg.TRAIN.LOG_DIR): shutil.rmtree(cfg.TRAIN.LOG_DIR)
writer = tf.summary.create_file_writer(cfg.TRAIN.LOG_DIR)

def train_step(model, image_data, target, epoch):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss=conf_loss=prob_loss=0

        # optimizing process
        for i in range(cfg.YOLO.BRANCH_SIZE):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> Epoch %2d    step %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" %(epoch,
                                                          global_steps,
                                                          total_steps,
                                                          optimizer.lr.numpy(),
                                                          giou_loss, conf_loss,
                                                          prob_loss, total_loss))
        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps *cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()

if __name__ == '__main__':
    model = get_model()
    # keras.utils.plot_model(model, "./model.png", show_shapes=True)
    if cfg.TRAIN.PRETRAIN is not None:
        model.load_weights(cfg.TRAIN.PRETRAIN)
        print('Restoring weights from: %s ... ' % cfg.TRAIN.PRETRAIN)

    img_write = True
    for epoch in range(cfg.TRAIN.CONTINUE_EPOCH, cfg.TRAIN.EPOCHS):
        for image_data, target in trainset:
            # 将第一个batch的图片写入TensorBoard
            if img_write is True:
                with writer.as_default():
                    tf.summary.image('Training data', image_data, max_outputs=10, step=0)
                img_write = False

            train_step(model, image_data, target, epoch)
        if epoch % 2 == 0:
            model.save_weights(cfg.TRAIN.CKPT_DIR+"model_epoch{}".format(epoch))
        # model.save('save/yolov3', save_format='tf')
    model.save_weights(cfg.TRAIN.CKPT_DIR + "model_final")
    model.save(cfg.TRAIN.CKPT_DIR+"saved_model", save_format='tf')

