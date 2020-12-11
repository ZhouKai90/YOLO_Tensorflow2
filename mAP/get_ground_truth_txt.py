import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from cfg.config import CFG


imageIDs = open(CFG.DATASET.IMAGESETS+'test.txt').read().strip().split()
inputPath = 'mAP/input'
GTPath = 'mAP/input/ground-truth/'

if not os.path.exists(inputPath):
    os.makedirs(inputPath)
if not os.path.exists(GTPath):
    os.makedirs(GTPath)

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
