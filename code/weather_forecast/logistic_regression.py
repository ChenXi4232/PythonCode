import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv("logisticdata.csv", header=0)  # 加载数据集
df.head()  # 预览前 5 行数据
print(df)

plt.figure(figsize=(10, 6))
plt.scatter(df['X0'], df['X1'], c=df['Y'])
plt.title('2017年合肥地区空气状态')
plt.show()


# Sigmoid 分布函数
def sigmoid(z):
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid


# 损失函数
def loss(h, y):
    loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    return loss


# 梯度计算
def gradient(X, h, y):
    gradient = np.dot(X.T, (h - y)) / y.shape[0]
    return gradient


# 逻辑回归过程
def Logistic_Regression(x, y, lr, num_iter):
    intercept = np.ones((x.shape[0], 1))  # 初始化截距为 1
    x = np.concatenate((intercept, x), axis=1)
    w = np.zeros(x.shape[1])  # 初始化参数为 0

    for i in range(num_iter):  # 梯度下降迭代
        z = np.dot(x, w)  # 线性函数
        h = sigmoid(z)  # sigmoid 函数

        g = gradient(x, h, y)  # 计算梯度
        w -= lr * g  # 通过学习率 lr 计算步长并执行梯度下降

        z = np.dot(x, w)  # 更新参数到原线性函数中
        h = sigmoid(z)  # 计算 sigmoid 函数值

        l = loss(h, y)  # 计算损失函数值

    return l, w  # 返回迭代后的梯度和参数


"""设置参数并训练得到结果
"""
import numpy as np

x = df[['X0', 'X1']].values
y = df['Y'].values
lr = 0.0003  # 学习率
num_iter = 3000  # 迭代次数

# 训练
L = Logistic_Regression(x, y, lr, num_iter)
print(L)
"""将上方得到的结果绘制成图
"""
plt.figure(figsize=(10, 6))
plt.scatter(df['X0'], df['X1'], c=df['Y'])

x1_min, x1_max = df['X0'].max(), df['X0'].min(),
x2_min, x2_max = df['X1'].min(), df['X1'].max(),

xx1, xx2 = np.meshgrid(np.linspace(x1_max, x1_min),
                       np.linspace(x2_max, x2_min))
grid = np.c_[xx1.ravel(), xx2.ravel()]

probs = (np.dot(grid, np.array([L[1][1:5]]).T) + L[1][0]).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, levels=[0], linewidths=1, colors='red')
"""绘制损失函数变化曲线
"""


def Logistic_Regression(x, y, lr, num_iter):
    intercept = np.ones((x.shape[0], 1))  # 初始化截距为 1
    x = np.concatenate((intercept, x), axis=1)
    w = np.zeros(x.shape[1])  # 初始化参数为 1

    l_list = []  # 保存损失函数值
    for i in range(num_iter):  # 梯度下降迭代
        z = np.dot(x, w)  # 线性函数
        h = sigmoid(z)  # sigmoid 函数

        g = gradient(x, h, y)  # 计算梯度
        w -= lr * g  # 通过学习率 lr 计算步长并执行梯度下降

        z = np.dot(x, w)  # 更新参数到原线性函数中
        h = sigmoid(z)  # 计算 sigmoid 函数值

        l = loss(h, y)  # 计算损失函数值
        l_list.append(l)

    return l_list


lr = 0.00005  # 学习率
num_iter = 5000  # 迭代次数
l_y = Logistic_Regression(x, y, lr, num_iter)  # 训练

# 绘图
plt.figure(figsize=(10, 6))
plt.plot([i for i in range(len(l_y))], l_y)
plt.title('空气质量状态逻辑回归')
plt.xlabel("迭代次数")
plt.ylabel("损失函数")
plt.show()