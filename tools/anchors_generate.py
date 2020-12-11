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


#***************************************anchors*****************************************************************
def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 10 or np.count_nonzero(y == 0) > 10:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def parse_anno(annotation_path, target_size=None):
    anno = open(annotation_path, 'r')
    result = []
    for line in anno:
        s = line.strip().split(' ')
        # print(s)
        img_w = int(s[1])
        img_h = int(s[2])
        s = s[3:]
        box_cnt = len(s) // 5
        for i in range(box_cnt):
            x_min, y_min, x_max, y_max = float(s[i*5]), float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3])
            width = x_max - x_min
            height = y_max - y_min
            assert width >= 0
            assert height >= 0
            # use letterbox resize, i.e. keep the original aspect ratio
            # get k-means anchors on the resized target image size
            if target_size is not None:
                resize_ratio = min(target_size[1] / img_w, target_size[0] / img_h)
                width *= resize_ratio
                height *= resize_ratio
                result.append([width, height])
            # get k-means anchors on the original image size
            else:
                result.append([width, height])
            # print([width, height])
    result = np.asarray(result)
    return result

def get_kmeans(anno, cluster_num=9):

    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])
    return anchors, ave_iou


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

    print("gen_TFAnno_txt finished.")
    anno_result = parse_anno(trainTFAnnoTxt, target_size=target_size)
    print("Total num for get kmeans: ", anno_result.shape)

    anchors, ave_iou = get_kmeans(anno_result, CFG.YOLO.BRANCH_SIZE*3)       #for tiny YOLO

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)

