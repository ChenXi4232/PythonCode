import pandas as pd

data = open(
    r'D:\科大\大一（下）\机器学习大作业-预测天气\数据\合肥3月11至3月17逐时气象资料\2022年 2月 蜀山区逐日.csv',
    encoding='utf-8')
pd.read_csv(data)
