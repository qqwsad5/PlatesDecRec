import cv2
import numpy as np
import os
from detection import getCorner,point_limit
import csv
import json

PATH = os.path.join(\
    os.path.split(os.path.realpath(__file__))[0], "../data/")

PATH_IMG = os.path.join(PATH,'img')
PATH_TRAIN = os.path.join(PATH,'train')

img_list = os.listdir(PATH_IMG)

IMG_PATH = 1
PLATE_NUM = 2
MODEL = 3
COLOR = 4
ULX = 6
ULY = 7
URX = 8
URY = 9
DRX = 10
DRY = 11
DLX = 12
DLY = 13

csv_reader = csv.reader(open(os.path.join(PATH,"label/annotation.csv"), encoding="utf-8"))

first = True

label_dic = {}

def point_trans(x,y):
    x = float(x)*512
    y = float(y)*512
    return [x,y]

def point_not_equal(a,b):
    if a[0]==b[0] and a[1]==b[1]:
        return False
    return True

w_out = 100
h_out = 30

for row in csv_reader:
    if first:
        first = False
        continue
    # assert row[IMG_PATH] in img_list, print(row)

    img_path = os.path.join(PATH_IMG,row[IMG_PATH])
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    p1 = point_trans(row[ULX],row[ULY])
    p2 = point_trans(row[URX],row[URY])
    p3 = point_trans(row[DLX],row[DLY])
    p4 = point_trans(row[DRX],row[DRY])

    points = np.array([[p1],[p2],[p3],[p4]])

    # 分别找到四角
    rd = getCorner(points,1,1)
    ru   = getCorner(points,1,-1)
    ld  = getCorner(points,-1,1)
    lu    = getCorner(points,-1,-1)

    if point_not_equal(rd,ru) and\
       point_not_equal(rd,ld) and\
       point_not_equal(rd,lu) and\
       point_not_equal(ru,ld) and\
       point_not_equal(ru,lu) and\
       point_not_equal(ld,lu):
       
        # 进行剪裁处理
        cor = np.array([lu, ru, rd, ld],dtype = "float32")

        lu = [cor[3][0], cor[3][1] - h_out]
        ld = [cor[3][0], cor[3][1]]
        ru = [cor[3][0] + w_out, cor[3][1] - h_out]
        rd = [cor[3][0] + w_out, cor[3][1]]

        cor_dst = np.array([lu, ru, rd, ld],dtype = "float32")

        per_trans = cv2.getPerspectiveTransform(cor, cor_dst)
        dst = cv2.warpPerspective(img, per_trans, (512, 512))

        point_limit(lu)
        point_limit(rd)

        dst = dst[int(lu[1]):int(rd[1]), int(lu[0]):int(rd[0])]

        # cv2.imwrite(os.path.join(PATH_TRAIN,row[IMG_PATH]), dst)

        # 中文路径存储
        # cv2.imencode('.jpg', dst)[1].tofile(os.path.join(PATH_TRAIN,row[IMG_PATH]))
        # exit()

    else:
        print(row)
        points = np.int0(points)
        img = cv2.drawContours(img, points, -1, (0, 0, 255), 5)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    # exit()


print('finish')

# # 将所有图片按照标签标注分割好
# for img in img_list:
