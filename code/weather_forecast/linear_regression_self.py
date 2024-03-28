# 训练所需库
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# 数据模型
import pandas as pd

# 绘图
import matplotlib.pyplot as plt

# 数值计算
import numpy as np

# 数据可视化
import seaborn as sns

from math import sqrt
import warnings

warnings.filterwarnings("ignore")


# 计时函数
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(),
                                 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\nTime taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))


train = pd.read_csv("D:\\Py_program\\code\\weather_forecast\\data.csv")
print(train.shape)  # 获得数组形状
train.info()  # 查看数组信息

plt.figure(figsize=(14, 7))  # 第一个子图
plt.subplot(1, 2, 1)
sns.countplot(x='RainToday', data=train)  # 第二个子图
plt.subplot(1, 2, 2)
sns.countplot(x='RainTomorrow', data=train)
# plt.show()

# 处理分类不平衡
no = train[train.RainTomorrow == 0]
yes = train[train.RainTomorrow == 1]
yes_oversampled = resample(yes,
                           replace=True,
                           n_samples=len(no),
                           random_state=123)  # 对yes进行过采样
train = pd.concat([no, yes_oversampled])  # 数据合并
fig = plt.figure(figsize=(8, 5))  # 绘图
train.RainTomorrow.value_counts(normalize=True).plot(kind='bar',
                                                     color=['skyblue', 'navy'],
                                                     alpha=0.9,
                                                     rot=0)
plt.title(
    'RainTomorrow Indicator No(0) and Yes(1) after Oversampling (Balanced Dataset)'
)
print(train.shape)  # 获得数组形状

# 将Date一列转换为日期时间格式
train['Date'] = pd.to_datetime(train['Date'], format='%Y%m%d')
train['year'] = train.Date.apply(
    lambda x: x.year)  # 将该列分为year, month, day三列, 并舍弃该列
train['month'] = train.Date.apply(lambda x: x.month)
train['day'] = train.Date.apply(lambda x: x.day)
train.drop(columns=['Date'], axis=1, inplace=True)
train.info()  # 预览变化

# 将所有字母变为小写
train = train.applymap(lambda s: s.lower() if type(s) == str else s)
train.columns = train.columns.str.strip().str.lower()
print(train)  # 预览变化

# 使用热图考察变量间相关性
fig, ax = plt.subplots(figsize=(20, 10))
mask = np.triu(np.ones_like(train.corr(), dtype=np.bool_))  # 隐藏上三角矩阵
sns.heatmap(train.corr(), annot=True, cmap="Reds", mask=mask,
            linewidth=0.5)  # annot = True -> 将数据写入格中

# 剔除相关系数在0.92以上的自变量
train.drop(columns=['surtemp', 'dewpoint', 'poevap', 'sunhours'],
           axis=1,
           inplace=True)
print(train)  # 预览变化

x = train.drop(['raintomorrow', 'temptomorrow'], axis=1)
y = train['raintomorrow']
z = train['temptomorrow']
print(x)  # 预览变化

# score_func = f_classif -> 评价特征方式为方差分析(Analysis of Variance, ANOA)
fs = SelectKBest(score_func=f_classif, k=14)
X_selected = fs.fit_transform(x, y)
print(X_selected.shape)
cols = fs.get_support(indices=True)  # 获取特征筛选结果
X_new = x.iloc[:, cols]  # 选取上述列的所有行
print(X_new)  # 预览变化

# 按照8：2划分训练集、测试集
trainX, testX, trainY, testY = train_test_split(X_new,
                                                y,
                                                test_size=0.2,
                                                random_state=999)
print(trainX.shape)
trainX.info()
print(testX.shape)
testX.info()
print(trainY.shape)
trainY.info()
print(testY.shape)
testY.info()

# 通过标准差标准化(standard scale)来进行数据的标准化
std_scaler = StandardScaler()  # 确定scaler
X_train = std_scaler.fit_transform(trainX)  # 训练并标准化
X_test = std_scaler.transform(testX)

# 逻辑回归
logRegTrain_cv = LogisticRegression()
# solver: 对逻辑回归损失函数的优化方法, 其中:
# newton-cg: 共轭梯度法, 利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数;
# lbfgs: 内存受限的 BFGS 算法, 用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数;
# liblinear: 使用了坐标轴下降法来迭代优化损失函数.
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l1', 'l2']  # penalty: 正则化选择参数
c_values = [100, 10, 1.0, 0.1, 0.01]  # C: 惩罚系数
grid = dict(solver=solvers, penalty=penalty, C=c_values)  # 网格搜索
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3,
                             random_state=1)  # k折交叉验证
grid_search = GridSearchCV(estimator=logRegTrain_cv,
                           param_grid=grid,
                           n_jobs=-1,
                           cv=cv,
                           scoring='accuracy',
                           error_score=0)
start_time = timer(None)  # 开始计时
grid_result = grid_search.fit(X_train, trainY)
timer(start_time)  # 停止计时
print("Best parameters: ", grid_result.best_params_)  # 结果
print("Model accuracy: ", grid_result.best_score_)

# 使用上述最佳参数
logRegTrain_grid = LogisticRegression(C=0.1, penalty="l2", solver='newton-cg')
logRegTrain_grid.fit(X_train, trainY)  # 训练模型
pred_grid = logRegTrain_grid.predict(X_test)  # 进行预测
# 评价指标
r2 = r2_score(pred_grid, testY)
LR_rmse = sqrt(mean_absolute_error(pred_grid, testY))
print("Mean squared error: %.4f" %
      metrics.mean_squared_error(pred_grid, testY))
print("Root mean absolute error: ", LR_rmse)
print("Mean absolute error: ", mean_absolute_error(pred_grid, testY))
print("R2 score: ", r2)
print("F1 score: ", f1_score(pred_grid, testY))
print("Training accuracy score: ", logRegTrain_grid.score(trainX, trainY))
print("Testing accuracy score: ", accuracy_score(pred_grid, testY))
print(classification_report(testY, pred_grid))

# XGBoost
params = {
    # min_child_weight: 最小叶子节点样本权重和, 用于避免过拟合
    'min_child_weight': [1, 5, 10],
    # gamma: 节点分裂所需的最小损失函数下降值, 值越大, 算法越保守
    'gamma': [0.5, 1, 1.5, 2, 5],
    # subsample: 对于每棵树，随机采样的比例, 值约小, 算法越保守, 避免过拟合
    'subsample': [0.6, 0.8, 1.0],
    # colsample_bytree: 用来控制每棵随机采样的列数的占比(每一列是一个特征)
    'colsample_bytree': [0.6, 0.8, 1.0],
    # max_depth: 树的最大深度, 值越大，模型会学到更具体更局部的样本
    'max_depth': [3, 4, 5]
}
xgb = XGBClassifier(learning_rate=0.02,
                    n_estimators=600,
                    objective='binary:logistic',
                    silent=True,
                    nthread=1)
folds = 5
param_comb = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
random_search = RandomizedSearchCV(xgb,
                                   param_distributions=params,
                                   n_iter=param_comb,
                                   scoring='roc_auc',
                                   n_jobs=4,
                                   cv=skf,
                                   verbose=3,
                                   random_state=1001)
start_time = timer(None)  # 开始计时
rf_result = random_search.fit(X_train, trainY)
timer(start_time)  # 停止计时
print("Best parameters: ", rf_result.best_params_)  # 结果
print("Model accuracy: ", rf_result.best_score_)

# 使用上述最佳参数
xgb = XGBClassifier(subsample=0.8,
                    min_child_weight=1,
                    max_depth=4,
                    gamma=1,
                    colsample_bytree=1.0)
xgb.fit(X_train, trainY)  # 训练模型
xgb_pred = xgb.predict(X_test)  # 进行预测
# 评价指标
xgb_r2 = r2_score(xgb_pred, testY)
xgb_rmse = sqrt(mean_absolute_error(xgb_pred, testY))
print("Mean squared error: %.4f" % metrics.mean_squared_error(xgb_pred, testY))
print("Root mean absolute error: ", xgb_rmse)
print("Mean absolute error: ", mean_absolute_error(xgb_pred, testY))
print("R2 score: ", xgb_r2)
print("F1 score: ", f1_score(xgb_pred, testY))
print("Testing accuracy score: ", accuracy_score(xgb_pred, testY))
print(classification_report(testY, xgb_pred))

# 随机森林
# 定义模型和参数
rfm = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
# 网格搜索
grid = dict(n_estimators=n_estimators, max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
rf_grid_search = GridSearchCV(estimator=rfm,
                              param_grid=grid,
                              n_jobs=-1,
                              cv=cv,
                              scoring='accuracy',
                              error_score=0)
start_time = timer(None)  # 开始计时
rf_grid_result = rf_grid_search.fit(X_train, trainY)  # 停止计时
timer(start_time)
print("Best parameters: ", rf_grid_result.best_params_)  # 结果
print("Model accuracy: ", rf_grid_result.best_score_)

# 使用上述最佳参数
rfm = RandomForestClassifier(n_estimators=1000, max_features='log2')
rfm.fit(X_train, trainY)  # 训练模型
rfm_pred = rfm.predict(X_test)  # 进行预测
# 评价指标
rfm_r2 = r2_score(rfm_pred, testY)
rfm_rmse = sqrt(mean_absolute_error(rfm_pred, testY))
print("Mean squared error: %.4f" % metrics.mean_squared_error(rfm_pred, testY))
print("Root mean absolute error: ", rfm_rmse)
print("Mean absolute error: ", mean_absolute_error(rfm_pred, testY))
print("R2 score: ", rfm_r2)
print("F1 score: ", f1_score(rfm_pred, testY))
print("Testing accuracy score: ", accuracy_score(rfm_pred, testY))
print(classification_report(testY, rfm_pred))

# 模型对比
accuracy_scores = [
    grid_result.best_score_, rf_result.best_score_, rf_grid_result.best_score_
]
model_data = {
    'Model': ['Logistic Regression', 'XGBoost', 'Random Forest'],
    'Accuracy': accuracy_scores
}
data = pd.DataFrame(model_data)
ax = data.plot.bar(x='Model', y='Accuracy', rot=0)
ax.set_title('Model Comparison: Accuracy for execution', fontsize=13)
plt.show()
