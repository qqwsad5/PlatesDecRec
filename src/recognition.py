import cv2
import numpy as np
import os
from numpy.linalg import norm
from detection import detection

PATH = os.path.join(\
    os.path.split(os.path.realpath(__file__))[0], "../data/img/0256.jpg")

# ==================================== 使用训练好的SVM模型 =======================================
PROVINCE_START = 1000
SZ = 20
#来自opencv的sample，用于svm训练
def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11']/m['mu02']
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img
#来自opencv的sample，用于svm训练
def preprocess_hog(digits):
	samples = []
	for img in digits:
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		bin_n = 16
		bin = np.int32(bin_n*ang/(2*np.pi))
		bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
		mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
		hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)
		
		# transform to Hellinger kernel
		eps = 1e-7
		hist /= hist.sum() + eps
		hist = np.sqrt(hist)
		hist /= norm(hist) + eps
		
		samples.append(hist)
	return np.float32(samples)
#不能保证包括所有省份
provinces = [
"zh_cuan", "川",
"zh_e", "鄂",
"zh_gan", "赣",
"zh_gan1", "甘",
"zh_gui", "贵",
"zh_gui1", "桂",
"zh_hei", "黑",
"zh_hu", "沪",
"zh_ji", "冀",
"zh_jin", "津",
"zh_jing", "京",
"zh_jl", "吉",
"zh_liao", "辽",
"zh_lu", "鲁",
"zh_meng", "蒙",
"zh_min", "闽",
"zh_ning", "宁",
"zh_qing", "靑",
"zh_qiong", "琼",
"zh_shan", "陕",
"zh_su", "苏",
"zh_sx", "晋",
"zh_wan", "皖",
"zh_xiang", "湘",
"zh_xin", "新",
"zh_yu", "豫",
"zh_yu1", "渝",
"zh_yue", "粤",
"zh_yun", "云",
"zh_zang", "藏",
"zh_zhe", "浙"
]
class StatModel(object):
	def load(self, fn):
		self.model = self.model.load(fn)  
	def save(self, fn):
		self.model.save(fn)
class SVM(StatModel):
	def __init__(self, C = 1, gamma = 0.5):
		self.model = cv2.ml.SVM_create()
		self.model.setGamma(gamma)
		self.model.setC(C)
		self.model.setKernel(cv2.ml.SVM_RBF)
		self.model.setType(cv2.ml.SVM_C_SVC)
#训练svm
	def train(self, samples, responses):
		self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
#字符识别
	def predict(self, samples):
		r = self.model.predict(samples)
		return r[1].ravel()
# ===============================================================================================

# 通过字符分割找到字符
def find_chars(histogram, d_thre, u_thre, max_thre):
    in_list = [[0, 100]]
    out_list = []
    threshold = 1
    while threshold <= max_thre:
        result = cut_hist(histogram, threshold, in_list)
        in_list = []
        for c in result:
            if c[1] - c[0] < d_thre: #则认为是点或者I,1等
                val = np.mean(histogram[c[0]:c[1]])
                if val > 10: #认为是I,1等，但还可能是开头和结尾的边缘
                    if c[0] > 10 and c[1] < 96:   #认为是I,1等，开头一定是汉字，结尾不可能靠边
                        out_list.append(c)
            elif c[1] - c[0] < u_thre: #则认为是可取的字符,长度为8\9\10\11\12\13
                out_list.append(c)
            else:   #认为需要继续切割
                in_list.append(c)
        if len(in_list) == 0: #没有需要继续切割的了
            break

        # 提高阈值，可以切割更宽的位置
        threshold += 1

    if len(in_list) != 0: #仍然需要继续切割
        char_size = (d_thre + u_thre - 1)/2
        for c in in_list:
            c_size = c[1] - c[0]
            cut_num = round(c_size / char_size)
            cut_result = cut_list(histogram, c, cut_num)
            for r in cut_result:
                out_list.append(r)

    # 按顺序输出
    out_list = sorted(out_list, key=lambda k: k[0])

    return out_list

#根据阈值分隔图片
def cut_hist(histogram, threshold, cut_list):
    out_list = []
    for i in cut_list:
        start = i[0]
        end = i[1]
        flag = False
        left = start
        while start<end:
            # 找开始
            if histogram[start]<threshold:
                if flag == True:  #找到的是结束点
                    flag = False
                    out_list.append([left,start])
            else:
                if flag == False: #找到的是开始点
                    flag = True
                    left = start
            start += 1

        if flag == True:    #证明已经开始
            out_list.append([left,end])
    
    return out_list

# 等分切割
def cut_list(histogram, cut_range, cut_num):
    out_list = []
    size = cut_range[1] - cut_range[0]
    cut_len = int(size / cut_num)
    cut_rest = size % cut_num
    pos = cut_range[0]
    for i in range(cut_rest):
        out_list.append([pos,pos+cut_len+1])
        pos += cut_len+1
    for i in range(cut_num - cut_rest):
        out_list.append([pos,pos+cut_len])
        pos += cut_len
    return out_list

# 切除上下边缘
def cut_edge(histogram):
    in_list = [[0, 30]]
    out_list = []
    threshold = 4
    while threshold <= 28:
        result = cut_hist(histogram, threshold, in_list)
        in_list = []
        for c in result:
            if c[1] - c[0] <= 5: #则认为是上下边缘
                pass
            elif c[1] - c[0] <= 22: #则认为可取了（最大成功宽度为22）
                out_list.append(c)
            else:   #认为需要继续切割
                in_list.append(c)
        if len(in_list) == 0: #没有需要继续切割的了
            break
        # 提高阈值，可以切割更宽的位置
        threshold += 4
    if len(in_list) != 0:   # 应该是切割失败了
        out_list.append(in_list[0])
    if len(out_list) == 0:   # 应该是完全切割失败了
        return [0, 30]

    return out_list[0]

def recognition(card_img, color):

    gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

    # 如果送过来的图片不满足要求
    if gray_img.shape[0] != 30 or gray_img.shape[1] != 100:
        gray_img = cv2.resize(gray_img, (100, 30), interpolation=cv2.INTER_AREA)

    #黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
    if color == "g" or color == "y":
        gray_img = cv2.bitwise_not(gray_img)

    # print(np.max(gray_img))
    # print(np.mean(gray_img))

    # threshold = int(np.max(gray_img) + np.mean(gray_img))/2

    ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imshow("color", gray_img)
    # cv2.waitKey(0)
    # exit()

    #直方图来定位各字符位置
    y_histogram  = np.sum(gray_img, axis=0)/255
    # print(y_histogram)

    x_histogram  = np.sum(gray_img, axis=1)/255
    # print(x_histogram)
    edges = cut_edge(x_histogram)
    # print(edges)

    if color == 'g':
        #绿牌切割参数
        chars = find_chars(y_histogram,8,13,5)
    else:
        # 蓝牌黄牌切割参数
        chars = find_chars(y_histogram,8,14,5)

    # print(chars)
    char_imgs = []
    h_img = edges[1] - edges[0]
    for c in chars:
        char_img = gray_img[edges[0]:edges[1], c[0]:c[1]]

        # 补成正方形
        w = abs(char_img.shape[1] - h_img)//2
        if w > 0:
            char_img = cv2.copyMakeBorder(char_img, 0, 0, w, w, cv2.BORDER_CONSTANT, value = [0,0,0])

        # 变成识别用的尺寸
        char_img = cv2.resize(char_img, (SZ, SZ), interpolation=cv2.INTER_AREA)

        char_imgs.append(char_img)
        # cv2.imshow("color", char_img)
        # cv2.waitKey(0)

    #识别英文字母和数字
    model = SVM(C=1, gamma=0.5)
    model.load("model/svm.dat")
    #识别中文
    modelchinese = SVM(C=1, gamma=0.5)
    modelchinese.load("model/svmchinese.dat")


    predict_result = []
    for i, char_img in enumerate(char_imgs):
        char_img = preprocess_hog([char_img])
        if i == 0:
            resp = modelchinese.predict(char_img)
            charactor = provinces[int(resp[0]) - PROVINCE_START]
            predict_result.append(charactor)
        else:
            resp = model.predict(char_img)
            charactor = chr(resp[0])
            predict_result.append(charactor)

    # print(predict_result)
    return predict_result
