# 导入数值计算库
import numpy as np
# 导入科学计算库
import pandas as pd
# 导入机器学习linear_model库
from sklearn import linear_model
# 导入交叉验证库
from sklearn import model_selection
# 导入图表库
import matplotlib.pyplot as plt
# 读取数据
datalist = pd.read_csv('lineardata.csv')
X = datalist.iloc[:, :1].values
Y = datalist.iloc[:, 1].values
print(datalist)
# 格式调整
X = np.array(datalist[['month']])  # 将月份数设为自变量X
Y = np.array(datalist['PM2.5'])  # PM2.5设为因变量Y
X.shape, Y.shape  # 查看自变量和因变量的行数

# 设置图表字体为华文细黑，字号11
plt.rc('font', family='STXihei', size=11)
# 绘制散点图，月份数X，PM2.5Y，设置颜色，标记点样式和透明度等参数
plt.scatter(X, Y, 30, color='red', marker='x', linewidth=2, alpha=0.8)
plt.xlabel('月份')  # 添加x轴标题
plt.ylabel('PM2.5值')  # 添加y轴标题
plt.title('2017年月份与PM2.5关系分析')  # 添加图表标题
# 设置背景网格线颜色，样式，尺寸和透明度
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='both', alpha=0.4)
plt.show()  # 显示图表

# 划分数据
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, Y, test_size=0.25, random_state=0)
# 查看训练集数据的行数
print('训练集的行数：')
print(X_train.shape, y_train.shape)
# 将训练集代入到线性回归模型中
clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)
clf.coef_  # 线性回归模型的斜率
clf.intercept_  # 线性回归模型的截距

# 判定系数R
clf.score(X_train, y_train)
print('判定系数R:')
print(clf.score(X_train, y_train))

# 显示测试集的因变量
print('测试集因变量：')
print(list(y_test))
# 将测试集的自变量代入到模型预测因变量
pred = list(clf.predict(X_test))
print('预测集因变量：')
print(pred)
# 训练结果的可视化
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, clf.predict(X_train), color='blue')
plt.show()
# 测试结果的可视化
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, clf.predict(X_test), color='green')
plt.show()
# 计算误差平方和
print('误差平方和：')
print(((y_test - clf.predict(X_test))**2).sum())
# 返回预测性能得分
print('Score:%.2f' % clf.score(X_test, y_test))
