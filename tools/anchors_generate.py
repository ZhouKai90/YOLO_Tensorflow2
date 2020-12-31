# coding: utf-8
from __future__ import division, print_function
from __future__ import division
import xml.etree.ElementTree as ET
import random
import os
import numpy as np
from cfg.config import CFG

def create_imagesets_txt(imgsetPath, annotationsPath, trainvalPercent=0.99):
    totalXml = os.listdir(annotationsPath)   # 获取标注文件（file_name.xml）
    # 训练数据集占总数据集的比例
    # print(trainvalPercent)
    tv = int(len(totalXml) * trainvalPercent)
    trainval = random.sample(range(len(totalXml)), tv)

    ftrainval   =   open(os.path.join(imgsetPath, 'trainval.txt'), 'w')
    ftest       =   open(os.path.join(imgsetPath, 'test.txt'), 'w')
    # ftrain      =   open(os.path.join(imgsetPath, 'train.txt'), 'w')
    # fval        =   open(os.path.join(imgsetPath, 'val.txt'), 'w')

    for i in range(len(totalXml)):                # 遍历所有 file_name.xml 文件
        name = totalXml[i][:-4] + '\n'            # 获取 file_name
        if i in trainval:
            ftrainval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftest.close()
    # ftrain.close()
    # fval.close()

def get_class_names(classNamesFile):
    namesDict = {}
    cnt = 0
    f = open(classNamesFile, 'r').readlines()
    for line in f:
        line = line.strip()
        # print(line)
        namesDict[line] = cnt
        cnt += 1
    return  namesDict

def parse_xml(path, namesDict):
    tree = ET.parse(path)
    img_name = path.split('/')[-1][:-4]

    height = tree.findtext("./size/height")
    width = tree.findtext("./size/width")

    objects = [img_name, width, height]

    for obj in tree.findall('object'):
        difficult = obj.find('difficult').text
        if difficult == '1':
            continue
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text

        name = str(namesDict[name])
        # print(name)
        objects.extend([xmin, ymin, xmax, ymax, name])
    if len(objects) > 3:
        return objects
    else:
        return None

def gen_TFAnno_txt(TFAnnoFile, annoPath, imgSetTxt, JPEGImgPath):
    f = open(TFAnnoFile, 'w')
    imgNames = open(imgSetTxt, 'r').readlines()
    for imgName in imgNames:
        imgName = imgName.strip()
        xmlPath = annoPath + '/' + imgName + '.xml'
        # objects = parse_xml(xmlPath, get_class_names(r'data/classes/pedestrian.names'))
        objects = parse_xml(xmlPath, get_class_names(CFG.YOLO.CLASSES))
        if objects:
            objects[0] = JPEGImgPath + '/' + imgName + '.jpg'
            if os.path.exists(objects[0]):
                objects = ' '.join(objects) + '\n'
                f.write(objects)
    f.close()

if __name__ == '__main__':
    # target resize format: [width, height]
    # if target_resize is speficied [416,416], the anchors are on the resized image scale
    # if target_resize is set to None, the anchors are on the original image scale

    target_size         =   CFG.TRAIN.INPUT_SIZE
    VOC2020             =   CFG.DATASET.PATH
    imageSetPath        =   VOC2020 + r'ImageSets/Main/'  # txt文件保存目录
    annotationsPath     =   VOC2020 + r'Annotations'
    JPEGImgPath         =   VOC2020 + r'JPEGImages'

    trainvalImgSetTxt   =   imageSetPath + r'trainval.txt'
    testImgSetTxt       =   imageSetPath + r'test.txt'

    nameclassFile       =    CFG.YOLO.CLASSES

    create_imagesets_txt(imageSetPath, annotationsPath, trainvalPercent=0.95)
    namesDir = get_class_names(nameclassFile)

    trainTFAnnoTxt = VOC2020 + 'trainval.txt'
    testTFAnnoTxt = VOC2020 + 'test.txt'
    gen_TFAnno_txt(trainTFAnnoTxt, annotationsPath, trainvalImgSetTxt, JPEGImgPath)
    gen_TFAnno_txt(testTFAnnoTxt, annotationsPath, testImgSetTxt, JPEGImgPath)


