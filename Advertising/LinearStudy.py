#-*- coding: utf-8 -*-
"""

Title:
author: xjzheng
date:

"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# read data
sales = pd.read_csv('Advertising.csv')
# # look data
# print(sales.head())
# print(sales.describe())

# 构造训练集和测试集
Train,Test = train_test_split(sales,train_size=0.8,random_state=1111)

# 建模
fit = smf.ols('sales~TV+radio+newspaper',data=Train).fit()
# 模型概览
# print(fit.summary())

# 重新建模
fit2 = smf.ols('sales~TV+radio',data=Train).fit()
# print(fit2.summary())

# 第一个模型的预测结果
pred = fit.predict(exog=Test)
# 第二个模型的预测结果
pred2 = fit2.predict(exog=Test.drop('newspaper',axis=1))

# 模型效果对比
RMSE = np.sqrt(mean_squared_error(Test.sales,pred))
RMSE2 = np.sqrt(mean_squared_error(Test.sales,pred2))
#
# print('第一个模型的预测效果：RMSE=%0.4f\n' %RMSE)
# print('第二个模型的预测效果：RMSE=%0.4f\n' %RMSE2)

# 真实值与预测值的关系# 设置绘图风格
plt.style.use('ggplot')
# 设置中文编码和负号的正常显示
plt.rcParams['font.sans-serif'] =['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 散点图
plt.scatter(Test.sales, pred, c='steelblue' ,label = '观测点')
# 回归线
plt.plot([Test.sales.min(), Test.sales.max()], [pred.min(), pred.max()], 'r--', lw=2, label = '拟合线')

# 添加轴标签和标题
plt.title('真实值VS.预测值')
plt.xlabel('真实值')
plt.ylabel('预测值')

# 去除图边框的顶部刻度和右边刻度
plt.tick_params(top = 'off', right = 'off')
# 添加图例
plt.legend(loc = 'upper left')
# 图形展现
plt.show()

