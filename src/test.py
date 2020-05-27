# from detection import detection
# from recognition import recognition
from baseline import detection
from baseline import recognition
import json
import time


fp = open('data/label/plate_num','r')
label = fp.read()
label = json.loads(label)
fp.close()


count_sum = 0

count_dic = []

start = time.time()

for path in label.keys():
    # 真实车牌
    # real_num = label[path]
    count_sum += 1

    # 车牌识别
    path = 'data/img/' + path
    card_imgs,colors = detection(path)
    if len(card_imgs) == 0:
        count_dic.append([])
        continue
    card_img = card_imgs[0]
    color = colors[0]

    # # 颜色转换
    # if color == 'b':
    #     color = 'blue'
    # elif color == 'g':
    #     color = 'green'
    # elif color == 'y':
    #     color = 'yello'


    try:
        recg_num = recognition(card_img, color)
    except:
        count_dic.append([])
        continue

    # count_dic.append(recg_num)

    # print(recg_num)
    # print(real_num)
    # exit()
    if count_sum == 100:
        break

    print ("\r {}".format(count_sum), end="")

elapsed = (time.time() - start)
print("Time: {}".format(elapsed))

# count_dic = json.dumps(count_dic)
# fp = open('data/label/my_base','w')
# fp.write(count_dic)
# fp.close()

# fp = open('data/label/base_base','r')
# count_dic = fp.read()
# count_dic = json.loads(count_dic)
# fp.close()

# count = 0
# count_true = 0
# count_sum = 0
# for path in label.keys():
#     # 真实车牌
#     real_num = label[path]
#     recg_num = count_dic[count]
#     # print(recg_num)
#     # print(real_num)
#     # exit()

#     count += 1


#     # 匹配正确数量
#     count_sum += len(real_num)
#     # num = min(len(real_num),len(recg_num))
#     num = len(recg_num)
#     for i in range(1,num):
#         # rn = real_num[i]

#         # # 模糊
#         # if rn == 'Q':
#         #     rn = '0'
#         # if rn == 'I':
#         #     rn = '1'
#         # if recg_num[i] == 'Q':
#         #     recg_num[i] = '0'
#         # if recg_num[i] == 'I':
#         #     recg_num[i] = '1'

#         if recg_num[i] in real_num:
#             count_true += 1

# print('\nrate:{}/{}({})'.format(count_true,count_sum,count_true/count_sum))