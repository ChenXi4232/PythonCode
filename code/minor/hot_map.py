import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("RS.txt", encoding='utf-8', delimiter="\t", thousands=',')

# 计算相关系数
correlation_matrix = data.corr()

# 绘制热图
plt.rcParams['font.family'] = 'SimSun'
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True,
            cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
