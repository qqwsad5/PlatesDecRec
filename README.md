# 车牌的检测和识别

## 主要包括了如下目录:

数据集的位置\
data/

SVM 模型的位置\
model/

代码位置\
src/\
其中包括了如下文件：\
UI 界面文件\
surface.py\
车牌检测文件\
detection.py\
车牌号码识别文件\
recognition.py

在根目录下执行 UI 界面文件，即可正确运行
（由于 CV2 的 SVM 模型加载好像不支持中文路径，因此文件中相关部分使用的都是相对路径，所以务必在根目录下运行才可正常运行）
```
python .\src\surface.py
```

