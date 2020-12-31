import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from cfg.config import CFG

inputPath = 'mAP/input'
GTPath = 'mAP/input/ground-truth/'
if not os.path.exists(inputPath):
    os.makedirs(inputPath)
if not os.path.exists(GTPath):
    os.makedirs(GTPath)

def get_voc_ground_truth():
    imageIDs = open(CFG.DATASET.IMAGESETS+'test.txt').read().strip().split()
    for imageID in tqdm(imageIDs):
        with open(GTPath+imageID+".txt", "w") as new_f:
            root = ET.parse(CFG.DATASET.ANNOTATIONS+imageID+".xml").getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult')!=None:
                    difficult = obj.find('difficult').text
                    if int(difficult)==1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text
                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

    print("Conversion completed!")

def get_odgt_ground_truth(class_name='head'):
    with open(CFG.DATASET.TEST_ANNOT, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    for annoLine in annotations:
        split_line = annoLine.split(" ")  # [name Xmin Ymin Xmax Ymax class] or [name W H Xmin Ymin Xmax Ymax class]
        image_name = split_line[0].split('/')[-1]
        # print("Reading {}".format(image_name))
        assert (len(split_line) - 3) % 5 == 0
        box_num = (len(split_line) - 3) / 5
        split_line = split_line[3:]
        with open(GTPath+image_name+".txt", "w") as new_f:
            for i in range(int(box_num)):
                box_xmin = int(float(split_line[i * 5]))
                box_ymin = int(float(split_line[i * 5 + 1]))
                box_xmax = int(float(split_line[i * 5 + 2]))
                box_ymax = int(float(split_line[i * 5 + 3]))
                new_f.write("%s %s %s %s %s\n" % (class_name, box_xmin, box_ymin, box_xmax, box_ymax))


if __name__ == '__main__':
    # get_voc_ground_truth()
    get_odgt_ground_truth(class_name='head')