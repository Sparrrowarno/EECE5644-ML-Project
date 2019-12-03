import os
import pandas as pd
import numpy as np
import shutil
from os import path

dir = path.dirname(__file__)
# dir =  "E:\\dnn\\distracted_driver_detection"

dir = os.path.join(dir, "data")
# dir拼接进如data文件夹
driver_imgs_list_csv = os.path.join(dir, "driver_imgs_list.csv")
# 创建csv表
valid_subjects_table = [['p002', 'p012'],
                        ['p014', 'p015'],
                        ['p016', 'p021'],
                        ['p024', 'p022'],
                        ['p035', 'p026'],
                        ['p042', 'p041', 'p039'],
                        ['p047', 'p045'],
                        ['p050', 'p049'],
                        ['p052', 'p051'],
                        ['p061', 'p056'],
                        ['p072', 'p066', 'p064'],
                        ['p081', 'p075']]
valid_indice = 9  # (0,1,2,3,4,5,6,7,8,9)
valid_subjects = valid_subjects_table[valid_indice]

if not os.path.exists(dir + "/imgs/valid"):
    os.mkdir(dir + "/imgs/valid")  # 确保data中有路径imgs/valid
    for i in range(10):
        os.mkdir(dir + "/imgs/valid/c%d" % i)  # 创建C0~C10

df = pd.read_csv(driver_imgs_list_csv)  # 读取driver_imgs_list_csv
for valid_subject in valid_subjects:
    df_valid = df[(df["subject"] == valid_subject)]
    for index, row in df_valid.iterrows():
        subpath = row["classname"] + "/" + row["img"]
        dirsub = os.path.join(dir, "train", subpath)
        if os.path.exists(os.path.join(dir, "imgs", "train", subpath)):
            shutil.move(os.path.join(dir, "imgs", "train", subpath), os.path.join(dir, "imgs", "valid", subpath), )
            print("move {} : {}".format(row["subject"], subpath))
        else:
            print("cannot move {} : {}".format(row["subject"], subpath))

