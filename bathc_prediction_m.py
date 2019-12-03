import os
import time
import shutil
from keras.models import *
from keras.layers import *
from keras_applications import *
from keras.preprocessing.image import *
from PIL import Image

#loading model
model0 = load_model(r"G:\project\distracted_driver_detection\models\VGG16.h5")
print("model0 load successed")
model1 = load_model(r"G:\project\distracted_driver_detection\models\InceptionV3.h5")
print("model0 load successed")
weight0 = [0.48235294, 0.475 ,0.46052632, 0.51388889, 0.5, 0.42253521, 0.34782609, 0.38271605, 0.44, 0.4516129 ]
weight1 = [0.51764706, 0.525 ,0.53947368, 0.48611111, 0.5, 0.57746479, 0.65217391, 0.61728395, 0.56,0.5483871 ]

# 批量预测
dir = "E:\\dnn\\distracted_driver_detection"
print(dir)
if not os.path.exists(dir + "/prediction"):
    os.mkdir(dir + "/prediction")  # 确保路径中有prediction文件夹
    for i in range(10):
        os.mkdir(dir + "/prediction/c%d" % i)  # 创建C0~C9


def combined_prediction_move(y_pred0, y_pred1, img_name):
    stats = [None] * 10
    for i in range(10):
        stats[i] = y_pred0[0, i] * weight0[i] + y_pred1[0, i] * weight1[i]
    pred_stat = stats.index(max(stats))
    shutil.copy("G:\\project\\distracted_driver_detection\\data\\imgs\\test\\" + img_name,
                "G:\\project\\distracted_driver_detection\\prediction\\c%d" % pred_stat)
    # shutil.copy("G:\\project\\distracted_driver_detection\\test_2\\"+img_name,  "G:\\project\\distracted_driver_detection\\prediction\\c%d"%pred_stat)
    return (pred_stat)


def final_prediction_move(y_pred, img_name):
    stats = [None] * 10
    for i in range(10):
        stats[i] = y_pred[0, i]
    pred_stat = stats.index(max(stats))
    shutil.copy("G:\\project\\distracted_driver_detection\\data\\imgs\\test\\" + img_name,
                "G:\\project\\distracted_driver_detection\\prediction\\c%d" % pred_stat)
    # shutil.copy("G:\\project\\distracted_driver_detection\\test_2\\"+img_name,  "G:\\project\\distracted_driver_detection\\prediction\\c%d"%pred_stat)
    return (pred_stat)


def final_prediction_possibility(y_pred):
    stats = [None] * 10
    for i in range(10):
        stats[i] = y_pred[0, i]
    acc = max(stats) / sum(stats)
    return (acc)


count = 0
pred_range1 = input("start from：")
pred_range2 = input("end in：")
pred_range1 = int(pred_range1)
pred_range2 = int(pred_range2) + 1
t0 = time.time()
for i in range(pred_range1, pred_range2):
    # print(img_name)
    img_name = str(i)
    img_name = "img_" + img_name + ".jpg"
    # img_name = input("图片文件号：")
    dir = r"G:\\project\\distracted_driver_detection\\data\\imgs\\test"
    # dir = r"G:\\project\\distracted_driver_detection\\test_2"
    dir = os.path.join(dir, img_name)

    if not os.path.exists(dir):
        print("无此图片")
        continue

    print(dir)
    img0 = Image.open(dir)
    img1 = Image.open(dir)
    t0 = time.time()

    img0 = image.load_img(dir, target_size=(224, 224))
    img1 = image.load_img(dir, target_size=(320, 480))

    x0 = image.img_to_array(img0)  # （224，224，3）
    x1 = image.img_to_array(img1)  # （224，224，3）
    x0 = np.expand_dims(x0, axis=0)  # （1，224，224，3）
    x1 = np.expand_dims(x1, axis=0)  # （1，224，224，3）
    # x = preprocess_input(x)
    y_pred0 = model0.predict(x0)
    y_pred1 = model1.predict(x1)
    # final_prediction_move(y_pred,img_name)
    combined__prediction_move(y_pred0, y_pred1, img_name)
    count += 1

t1 = time.time()
print("done!")
print("总数量", count)