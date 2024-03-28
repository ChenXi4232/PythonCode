import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

# read csv


def read_csv_pd(filename):
    data = pd.read_csv(filename, header=None)
    return data


dir_path = os.path.dirname(os.path.realpath(__file__))
format_dir_path = dir_path + "\\format_data\\"
raw_dir_path = dir_path + "\\raw_data\\"
# 读取指定路径下的所有csv文件并处理
for file in os.listdir(raw_dir_path):
    print(file)
    data = read_csv_pd(raw_dir_path + file)
    print(data.shape)
    # 删掉文档中指定行并保存
    data = data.drop([0, 1, 2, 5, 6, 7, 8, 9, 10])
    try:
        data.to_csv(format_dir_path + file, index=False, header=False)
        print(data.shape)
    # 删除 file 路径的文件
        file_path = os.path.join(raw_dir_path, file)
        os.remove(file_path)
    except Exception as e:
        # 将报错写入文件
        with open(dir_path + "\\error.txt", 'a') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S',
                    time.localtime(time.time())) + str(e) + "\n")
        print(e)
        continue
