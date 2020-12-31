# coding: utf-8
import time
from PIL import Image
import os
import json
from tqdm import *
import random
import cv2

separate = ' '

def load_file(fpath):  # fpath是具体的文件 ，作用：#str to list
    assert os.path.exists(fpath)  # assert() raise-if-not
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]  # str to list
    return records


def img2txt(imgPath, odgtpath, respath):
    records = load_file(odgtpath)  # 提取odgt文件数据
    record_list = len(records)  # 获得record的长度，循环遍历所有数据。
    print(os.getcwd())
    # os.mkdir(os.getcwd() + respath)
    with open(respath, 'w') as txt:
        for i in range(record_list):
            file_name = records[i]['ID'] + '.jpg'
            file_name = str(imgPath + file_name)
            txt.write(file_name + '\n')


def tonormlabel(imgPath, odgtpath, storeFile, attr='hbox'):
    file = open(storeFile, 'w')
    records = load_file(odgtpath)
    record_list = len(records)
    print('Total num:', record_list)

    for i in tqdm(range(record_list)):
        file_name = records[i]['ID'] + '.jpg'
        gt_box = records[i]['gtboxes']
        gt_box_len = len(gt_box)  # 每一个字典gtboxes里，也有好几个记录，分别提取记录。
        if gt_box_len == 0:
            continue
        im = Image.open(imgPath + file_name)
        height = im.size[1]
        width = im.size[0]
        str_to_write = imgPath + file_name + separate + str(width) + separate + str(height)
        label_used = False

        for j in range(gt_box_len):
            category = gt_box[j]['tag']
            if category != 'person':  # 该类型不是人，其中有mask标签
                continue
            if attr == 'hbox':
                head_attr = gt_box[j]['head_attr']
                if len(head_attr) == 0:
                    continue
                if int(head_attr['ignore']) != 0 or int(head_attr['unsure']) != 0:   #忽略，遮挡，不确定的人头标记，跳过忽略
                    # int(head_attr['occ']) != 0 or \
                    continue

            fbox = gt_box[j][attr]  # 获得框 [x_min, y_min, w, h]
            x_min = int(fbox[0])
            y_min = int(fbox[1])
            x_max = int(fbox[0] + fbox[2])
            y_max = int(fbox[1] + fbox[3])


            x_min = 0 if x_min <= 0 else x_min
            x_min = width if x_min >= width else x_min

            y_min = 0 if y_min <= 0 else y_min
            y_min = height if y_min >= height else y_min

            x_max = 0 if x_max <= 0 else x_max
            x_max = width if x_max >= width else x_max

            y_max = 0 if y_max <= 0 else y_max
            y_max = height if y_max >= height else y_max

            if y_max - y_min <= 0 or x_max - x_min <= 0:
                continue

            str_to_write += separate + str(x_min) + separate + str(y_min) + separate \
                            + str(x_max) + separate + str(y_max) + separate + str(0)
            label_used = True       #确保标记出来的框至少有一个能用
        if label_used:
            file.write(str_to_write + '\n')
        else:
            print(imgPath + file_name)
    file.close()

def create_train_val_imagesets(imgsetPath, trainvalAnnotationsFile, prefix, trainvalPercent=0.9):
    # 训练数据集占总数据集的比例
    with open(trainvalAnnotationsFile, 'r') as fid:
        annos = fid.readlines()

    random.shuffle(annos)
    trainNum = int(len(annos) * trainvalPercent)
    print('Total num:', len(annos))
    print('Train num:', trainNum)
    print('Val num: ', len(annos) - trainNum)

    with open(os.path.join(imgsetPath, prefix+'_train.txt'), 'w') as ftrain:
        for i in range(trainNum):
            ftrain.write(annos[i])

    with open(os.path.join(imgsetPath, prefix+'_val.txt'), 'w') as fval:
        for i in range(trainNum, len(annos)):
            fval.write(annos[i])

'''将标注格式修改为 img_path width, height, x1, y1, x2, y2, class_id x1, y1, x2, y2 classid .....'''
if __name__ == '__main__':
    TRAINVALJPEGIMGPATN = 'data/dataset/crowd_human/trainval/'
    trainvalOdgtPath = "data/dataset/crowd_human/annotation_trainval.odgt"
    annoPath = "data/dataset/crowd_human/"
    trainvalAnnoFile = annoPath + "person_trainval.txt"
    tonormlabel(TRAINVALJPEGIMGPATN, trainvalOdgtPath, trainvalAnnoFile, attr='fbox')
    create_train_val_imagesets(annoPath, trainvalAnnoFile, prefix='person', trainvalPercent=0.95 )
