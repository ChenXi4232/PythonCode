import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 13, 1)
y = np.array([
    85.67741935, 84.42857143, 61.19354839, 50.7, 39.41935484, 42.16666667,
    31.35483871, 25.80645161, 34.83333333, 40.2, 64.66666667, 91.35483871
])
z1 = np.polyfit(x, y, 8)  # 用2次多项式拟合
p1 = np.poly1d(z1)
print("拟合多项式为：")
print(p1)  # 在屏幕上打印拟合多项式
yvals = p1(x)  # 也可以使用yvals=np.polyval(z1,x)
plot1 = plt.plot(x, y, '*', label='truth pm2.5')
plot2 = plt.plot(x, yvals, 'r', label='fitting pm2.5')
plt.xlabel('month')
plt.ylabel('PM2.5')
plt.legend(loc=4)  # 指定legend的位置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title(u'2017年合肥月份与PM2.5值关系')
plt.show()
