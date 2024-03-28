# 训练所需库
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 数据模型
import pandas as pd

# 绘图
import matplotlib.pyplot as plt

# 数值计算
import numpy as np

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


data = np.loadtxt('D:\\Py_program\\code\\posture_recognition\\X_train.txt')
print(data.shape)  # 获得数组形状
train = pd.DataFrame(data)
train.info()  # 查看数组信息
print(train)

trainX = train
trainY = np.loadtxt('D:\\Py_program\\code\\posture_recognition\\y_train.txt')
testX = np.loadtxt('D:\\Py_program\\code\\posture_recognition\\X_test.txt')
testX = pd.DataFrame(testX)
testY = np.loadtxt('D:\\Py_program\\code\\posture_recognition\\y_test.txt')

le = LabelEncoder()
print(trainY)
trainY = le.fit_transform(trainY)
trainY = pd.DataFrame(trainY)
print(trainY)
testY = le.fit_transform(testY)
testY = pd.DataFrame(testY)

print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

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
                                   n_jobs=4,
                                   cv=skf,
                                   verbose=3,
                                   random_state=1001)
start_time = timer(None)  # 开始计时
rf_result = random_search.fit(trainX, trainY)
timer(start_time)  # 停止计时
print("Best parameters: ", rf_result.best_params_)  # 结果
print("Model accuracy: ", rf_result.best_score_)

# 使用上述最佳参数
xgb = XGBClassifier(subsample=1.0,
                    min_child_weight=5,
                    max_depth=3,
                    gamma=5,
                    colsample_bytree=1.0)
xgb.fit(trainX, trainY)  # 训练模型
xgb_pred = xgb.predict(testX)  # 进行预测
# 评价指标
xgb_r2 = r2_score(xgb_pred, testY)
xgb_rmse = sqrt(mean_absolute_error(xgb_pred, testY))
print("Mean squared error: %.4f" % metrics.mean_squared_error(xgb_pred, testY))
print("Root mean absolute error: ", xgb_rmse)
print("Mean absolute error: ", mean_absolute_error(xgb_pred, testY))
print("R2 score: ", xgb_r2)
print("F1 score: ", f1_score(xgb_pred, testY, average='micro'))
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
                              error_score=0)
start_time = timer(None)  # 开始计时
rf_grid_result = rf_grid_search.fit(trainX, trainY)  # 停止计时
timer(start_time)
print("Best parameters: ", rf_grid_result.best_params_)  # 结果
print("Model accuracy: ", rf_grid_result.best_score_)

# 使用上述最佳参数
rfm = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
rfm.fit(trainX, trainY)  # 训练模型
rfm_pred = rfm.predict(testX)  # 进行预测
# 评价指标
rfm_r2 = r2_score(rfm_pred, testY)
rfm_rmse = sqrt(mean_absolute_error(rfm_pred, testY))
print("Mean squared error: %.4f" % metrics.mean_squared_error(rfm_pred, testY))
print("Root mean absolute error: ", rfm_rmse)
print("Mean absolute error: ", mean_absolute_error(rfm_pred, testY))
print("R2 score: ", rfm_r2)
print("F1 score: ", f1_score(rfm_pred, testY, average='micro'))
print("Testing accuracy score: ", accuracy_score(rfm_pred, testY))
print(classification_report(testY, rfm_pred))

# 模型对比
accuracy_scores = [
    rf_result.best_score_, rf_grid_result.best_score_
]
model_data = {
    'Model': ['XGBoost', 'Random Forest'],
    'Accuracy': accuracy_scores
}
data = pd.DataFrame(model_data)
ax = data.plot.bar(x='Model', y='Accuracy', rot=0)
ax.set_title('Model Comparison: Accuracy for execution', fontsize=13)
plt.show()
