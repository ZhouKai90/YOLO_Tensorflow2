import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from cfg.config import CFG

# imageIDs = open(CFG.DATASET.IMAGESETS+'test.txt').read().strip().split()
imageIDs = open(CFG.DATASET.IMAGESETS+'trainval.txt').read().strip().split()


for imageID in tqdm(imageIDs ):
    xml = ET.parse(CFG.DATASET.ANNOTATIONS+imageID+".xml")
    root = xml.getroot()
    for obj in root.findall('object'):
        obj_name = obj.find('name')
        objRename = obj_name.text.replace(' ', '_')
        obj_name.text = objRename
        # obj.set('name', objRename)
    xml.write(CFG.DATASET.ANNOTATIONS+imageID+".xml", 'utf-8', True)
print("Conversion completed!")