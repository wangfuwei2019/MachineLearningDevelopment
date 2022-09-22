import sklearn.linear_model as lm

# 人工生成学习数据，包括正确答案（正确标签）
X = [[11.0], [9.0], [13.1], [9.2], [11.0], [13.9], [6.1], [4.0], [11.8], [7.1], [5.3]]
y = [8.05, 6.99, 7.57, 8.82, 8.34, 9.95, 7.25, 4.27, 10.90, 4.81, 5.69]

# 建模
model = lm.LinearRegression()
# 学习
model.fit(X, y)

print(model.coef_)          # 相关系数
print(model.intercept_)     # 截距

# 可以一次性预测一个或多个值
y_predict = model.predict([[15], [16]])
print('prediction value is:', y_predict)
