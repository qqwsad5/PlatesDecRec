import cv2
import numpy as np
import os
from detection import detection

PATH = os.path.join(\
    os.path.split(os.path.realpath(__file__))[0], "../data/img/0000.jpg")

rois = detection(PATH)
for roi in rois:
    cv2.imshow("color", roi)
    cv2.waitKey(0)

def find_waves(threshold, histogram):
	up_point = -1#上升点
	is_peak = False
	if histogram[0] > threshold:
		up_point = 0
		is_peak = True
	wave_peaks = []
	for i,x in enumerate(histogram):
		if is_peak and x < threshold:
			if i - up_point > 2:
				is_peak = False
				wave_peaks.append((up_point, i))
		elif not is_peak and x >= threshold:
			is_peak = True
			up_point = i
	if is_peak and up_point != -1 and i - up_point > 4:
		wave_peaks.append((up_point, i))
	return wave_peaks

#根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
	part_cards = []
	for wave in waves:
		part_cards.append(img[:, wave[0]:wave[1]])
	return part_cards

card_img = rois[0]
gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
#黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向

ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#查找水平直方图波峰
x_histogram  = np.sum(gray_img, axis=1)
x_min = np.min(x_histogram)
x_average = np.sum(x_histogram)/x_histogram.shape[0]
x_threshold = (x_min + x_average)/2
wave_peaks = find_waves(x_threshold, x_histogram)
if len(wave_peaks) == 0:
    print("peak less 0:")
    exit()
#认为水平方向，最大的波峰为车牌区域
wave = max(wave_peaks, key=lambda x:x[1]-x[0])
gray_img = gray_img[wave[0]:wave[1]]
#查找垂直直方图波峰
row_num, col_num= gray_img.shape[:2]
#去掉车牌上下边缘1个像素，避免白边影响阈值判断
gray_img = gray_img[1:row_num-1]
y_histogram = np.sum(gray_img, axis=0)
y_min = np.min(y_histogram)
y_average = np.sum(y_histogram)/y_histogram.shape[0]
y_threshold = (y_min + y_average)/5#U和0要求阈值偏小，否则U和0会被分成两半

wave_peaks = find_waves(y_threshold, y_histogram)

#for wave in wave_peaks:
#	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2) 
#车牌字符数应大于6
if len(wave_peaks) <= 6:
    print("peak less 1:", len(wave_peaks))
    exit()

wave = max(wave_peaks, key=lambda x:x[1]-x[0])
max_wave_dis = wave[1] - wave[0]
#判断是否是左侧车牌边缘
if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis/3 and wave_peaks[0][0] == 0:
    wave_peaks.pop(0)

#组合分离汉字
cur_dis = 0
for i,wave in enumerate(wave_peaks):
    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
        break
    else:
        cur_dis += wave[1] - wave[0]
if i > 0:
    wave = (wave_peaks[0][0], wave_peaks[i][1])
    wave_peaks = wave_peaks[i+1:]
    wave_peaks.insert(0, wave)

#去除车牌上的分隔点
point = wave_peaks[2]
if point[1] - point[0] < max_wave_dis/3:
    point_img = gray_img[:,point[0]:point[1]]
    if np.mean(point_img) < 255/5:
        wave_peaks.pop(2)

if len(wave_peaks) <= 6:
    print("peak less 2:", len(wave_peaks))
    exit()
part_cards = seperate_card(gray_img, wave_peaks)

SZ = 20          #训练图片长宽

for i, part_card in enumerate(part_cards):
    #可能是固定车牌的铆钉
    if np.mean(part_card) < 255/5:
        print("a point")
        continue
    part_card_old = part_card
    w = abs(part_card.shape[1] - SZ)//2
    
    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value = [0,0,0])
    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
    cv2.imshow("color", part_card)
    cv2.waitKey(0)