import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 定义你要拟合的曲线函数，例如：二次函数


def log_func(x, a, b, c):
    return a * np.log(b * x) + c


def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c


def power_func(x, a, b, c):
    return a * np.power(x, b) + c


def model_func(x, a, b, c):
    return a * x**2 + b * x + c


# 示例数据，多组数据，每组数据有 x 和 y 值
data_groups = [
    (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
     np.array([9516951.38, 9926809.32,
               10479522.57,               11303501.29,
               12120385.01,
               13635209.52,
               15189084.03,
               16655739.41,
               18664126.39
               ])),
    (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
     np.array([9516951.38, 9867031.15,
               10321156.86,
               11146597.82,
               12109125.89,
               13420892.98,
               14932053.73,
               16408575.89,
               18115055.60
               ])),
    (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
     np.array([9516951.38, 9899837.40,
               10367249.82,
               10974466.57,
               11979890.34,
               13273901.07,
               14723249.69,
               16308071.62,
               18212466.49,
               ])),
]

# 存储所有参数的列表
params_list = []

# 存储所有拟合结果的误差
errors = []

# 对每组数据进行曲线拟合
for x_data, y_data in data_groups:
    # 使用 curve_fit 函数进行拟合
    popt, pcov = curve_fit(power_func, x_data, y_data)
    params_list.append(popt)

    # 计算拟合结果
    y_fit = power_func(x_data, *popt)

    # 计算拟合误差（均方误差）
    error = np.mean((y_data - y_fit) ** 2)
    errors.append(error)

    # 绘制每组数据的拟合曲线
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, y_fit, label='Fit: a=%5.3f, b=%5.3f, c=%5.3f' %
             tuple(popt))
    plt.legend()
    plt.show()

# 计算所有组的平均误差
average_error = np.mean(errors)
print("Average Mean Squared Error:", average_error)

# 计算平均参数值
average_params = np.mean(params_list, axis=0)
print("Average Parameters:", average_params)

# 打印平均参数值
print(
    f'Average parameters: a={average_params[0]}, b={average_params[1]}, c={average_params[2]}')
