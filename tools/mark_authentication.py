import cv2
import os
if __name__ == '__main__':
    annotation_path = r'data/dataset/crowd_human/person_train.txt'
    assert os.path.exists(annotation_path)  # assert() raise-if-not
    with open(annotation_path, 'r') as fid:
        lines = fid.readlines()
    anno = [(line.strip('\n')) for line in lines]  # str to list
    print(len(anno))
    for n, line in enumerate(anno):
        s = line.strip().split(' ')
        # print(s)
        name = s[0]
        img_w = int(s[1])
        img_h = int(s[2])
        s = s[3:]
        assert len(s) % 5 == 0
        box_cnt = len(s) // 5
        mat = cv2.imread(name)
        bbox_thick = int(0.6 * (img_h + img_w) / 600)
        for i in range(box_cnt):
            x_min, y_min, x_max, y_max = int(s[i*5]), int(s[i*5+1]), int(s[i*5+2]), int(s[i*5+3])
            cv2.rectangle(mat, (x_min, y_min), (x_max, y_max), (0 , 0, 255), bbox_thick)
        cv2.imwrite('tools/out/'+name.split('/')[-1], mat)
        if n > 10:
            break
