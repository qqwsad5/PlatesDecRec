import cv2
import numpy as np
import os

Min_Area = 1200  #车牌区域允许最小面积

PATH = os.path.join(\
    os.path.split(os.path.realpath(__file__))[0], "../data/img/0001.jpg")

def point_limit(point):
	if point[0] < 0:
		point[0] = 0
	if point[1] < 0:
		point[1] = 0

def getCorner(points, dir0, dir1):
    result = [-99999*dir0,-99999*dir1]
    for p in points:
        p = p[0]
        if p[0]*dir0 + p[1]*dir1 > result[0]*dir0 + result[1]*dir1:
            result[0] = p[0]
            result[1] = p[1]
    return result

def accurate_place(card_img_hsv, color):
    row_num, col_num = card_img_hsv.shape[:2]

    img = np.zeros((row_num,col_num), np.uint8)

    # 二值化
    for row in range(row_num):
        for col in range(col_num):
            H,S,V = card_img_hsv[row,col,:]
            if getColor(H,S,V) == color:
                img[row,col] = 255

    # cv2.imshow('img', img)

    img_edge2 = img

    #查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    try:
        contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours)==0:
        return []
    max_cnt = max(contours, key=lambda i: cv2.contourArea(i))

    # 再做一次大小阈值筛选
    if cv2.contourArea(max_cnt) < Min_Area:
        return []

    # 再做一次长宽比筛选
    rect = cv2.minAreaRect(max_cnt)
    area_width, area_height = rect[1]
    if area_width < area_height:
        area_width, area_height = area_height, area_width
    wh_ratio = area_width / area_height
    #要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
    if wh_ratio <= 2 or wh_ratio >= 5.5:
        return []

    # 分别找到四角
    right_down = getCorner(max_cnt,1,1)
    right_up   = getCorner(max_cnt,1,-1)
    left_down  = getCorner(max_cnt,-1,1)
    left_up    = getCorner(max_cnt,-1,-1)

    rate = 0.2
    # 绿色还要多取一点
    if color == 'g':
        right_up[0] += int(rate*(right_up[0] - right_down[0]))
        right_up[1] += int(rate*(right_up[1] - right_down[1]))
        left_up[0]  += int(rate*(left_up[0]  - left_down[0]))
        left_up[1]  += int(rate*(left_up[1]  - left_down[1]))

    cor = np.array([left_up, right_up, right_down, left_down],dtype = "float32")

    return cor

def getColor(H,S,V):
    if 11 < H <= 34 and S > 40 and V > 45:#图片分辨率调整
        return 'y'
    elif 35 < H <= 99 and S > 40 and V > 45:#图片分辨率调整
        return 'g'
    elif 99 < H <= 124 and S > 40 and V > 45:#图片分辨率调整
        return 'b'
    # if 0 < H <180 and 0 < S < 255 and 0 < V < 46:
    #     return 'black'
    elif 0 < H <180 and 0 < S < 43 and 150 < V < 225:
        return 'w'
    
    return 'n'


def detection_img(img_in):

    img = img_in

    pic_hight, pic_width = img.shape[:2]


    #高斯去噪
    img = cv2.GaussianBlur(img, (3, 3), 0)
    oldimg = img

    # 转灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #找到图像边缘
    img_edge = cv2.Canny(img, 200, 250)

    # cv2.imshow('img_edge', img_edge)

    #使用开运算让图像边缘成为一个整体
    kernel = np.ones((2, 20), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow('img_edge1', img_edge1)

    #使用闭运算除掉孤立点
    kernel = np.ones((15, 15), np.uint8)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

    # cv2.imshow('img_edge2', img_edge2)

    # cv2.waitKey(0)
    # exit()

    #查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    try:
        contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
    print('len(contours)', len(contours))

    #一一排除不是车牌的矩形区域
    car_contours = []
    for cnt in contours:
        
        rect = cv2.minAreaRect(cnt)

        area_width, area_height = rect[1]
        if area_width < area_height:
            area_width, area_height = area_height, area_width
        wh_ratio = area_width / area_height
        #print(wh_ratio)
        #要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
        if wh_ratio > 2 and wh_ratio < 5.5:

            rect = (rect[0], (rect[1][0]+20, rect[1][1]+40), rect[2]) #扩大范围

            car_contours.append(rect)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)

    print(len(car_contours))
    # cv2.imshow('img', oldimg)
    # cv2.waitKey(0)
    # exit()

    print("精确定位")

    card_imgs = []
    #将矩形旋转正来使用
    for rect in car_contours:

        # print(rect)

        rot_mat = cv2.getRotationMatrix2D(rect[0],rect[2],1)

        rotated_img = cv2.warpAffine(oldimg, rot_mat, (oldimg.shape[1], oldimg.shape[0]))

        l = [rect[0][0] - 0.5*rect[1][0], rect[0][1] - 0.5*rect[1][1]]
        r = [rect[0][0] + 0.5*rect[1][0], rect[0][1] + 0.5*rect[1][1]]

        point_limit(l)
        point_limit(r)

        card_img = rotated_img[int(l[1]):int(r[1]), int(l[0]):int(r[0])]

        # 图像旋转90度，变为正对
        if rect[2]<-45:
            card_img = cv2.transpose(card_img)
            card_img = cv2.flip(card_img, 0)

        card_imgs.append(card_img)

        # cv2.imshow('img', card_img)
        # cv2.waitKey(0)


    #开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
    out_imgs = []
    for card_index,card_img in enumerate(card_imgs):
        green = yello = blue = black = white = 0
        card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        #有转换失败的可能，原因来自于上面矫正矩形出错
        if card_img_hsv is None:
            continue
        row_num, col_num= card_img_hsv.shape[:2]
        card_img_count = row_num * col_num

        for i in range(row_num):
            for j in range(col_num):
                H,S,V = card_img_hsv[i,j,:]
                hsv_color = getColor(H,S,V)
                if hsv_color == 'y':
                    yello += 1
                elif hsv_color == 'g':
                    green += 1
                elif hsv_color == 'b':
                    blue += 1
                elif hsv_color == 'w':
                    white += 1

        color = "no"

        # 由于前面剪切范围较大，更改阈值
        limit1 = limit2 = 0
        rate = 2
        while limit1==0 and rate<=6:
            if blue*rate >= card_img_count:
                color = "b"
            elif yello*rate >= card_img_count:
                color = "y"
            elif int(green*rate*1.5) >= card_img_count:
                color = "g"
            # 降低要求比率
            rate = rate + 1
            # elif black + white >= card_img_count*0.7:#TODO
            #     color = "bw"

        print(color)
        print(blue, green, yello, black, white, card_img_count)

        w_out = 100
        h_out = 30
        if color != 'no':
            row_size, col_size= card_img.shape[:2]

            cor = accurate_place(card_img_hsv, color)
            if len(cor) == 0:
                continue

            lu = [cor[3][0], cor[3][1] - h_out]
            ld = [cor[3][0], cor[3][1]]
            ru = [cor[3][0] + w_out, cor[3][1] - h_out]
            rd = [cor[3][0] + w_out, cor[3][1]]

            cor_dst = np.array([lu, ru, rd, ld],dtype = "float32")

            per_trans = cv2.getPerspectiveTransform(cor, cor_dst)
            dst = cv2.warpPerspective(card_img, per_trans, (col_size, row_size))

            point_limit(lu)
            point_limit(rd)

            out_imgs.append(dst[int(lu[1]):int(rd[1]), int(lu[0]):int(rd[0])])
            # cv2.imshow("color", dst)
            # cv2.waitKey(0)
            # exit()

    return out_imgs


def detection(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    rol, col = img.shape[:2]
    center = (rol/2,col/2)
    rois = detection_img(img)
    if len(rois) == 0:
        rot_M = cv2.getRotationMatrix2D(center, 20, 1)
        rotated_img = cv2.warpAffine(img, rot_M, (col, rol))
        rois = detection_img(rotated_img)
    if len(rois) == 0:
        rot_M = cv2.getRotationMatrix2D(center, -20, 1)
        rotated_img = cv2.warpAffine(img, rot_M, (col, rol))
        rois = detection_img(rotated_img)

    return rois


# rois = detection(PATH)
# for roi in rois:
#     cv2.imshow("color", roi)
#     cv2.waitKey(0)