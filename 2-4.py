import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

# 函数定义
def linearRegressionPredict():
    x = np.random.rand(300, 1)       # 生成符合均匀分布值域范围在【0，1】的数据
    x = x*6 - 3                      # 对x进行约束，使其值域范围控制在【-3，3】之间
    y = 4*x + 3                      # 对y进行约束，使其值域范围控制在【-9，15】之间
    y = y + np.random.randn(300, 1)  # 将y值增加噪声，生成300个符合标准正态分布的噪声值

    plt.scatter(x, y, marker='.')    # 用散点图显示人工生成带噪声的数据

    model = lm.LinearRegression()    # 线性回归模型实例化
    model.fit(x, y)                  # 将数据输入模型进行学习

    # 用散点图显示预测值函数图像
    plt.scatter(x, model.predict(x), marker='*')
    plt.show()


# 函数执行
linearRegressionPredict()

