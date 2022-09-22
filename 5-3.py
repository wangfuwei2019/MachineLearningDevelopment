import numpy as np

# 数据初始化
data = 1200  # 数据数量
X = np.zeros((data, 2))
T = np.zeros((data))  # 正确答案

for i in range(data):
    # 随机设定x、y坐标
    x_random = np.random.rand()  # x坐标
    y_random = np.random.rand()  # y坐标
    X[i, 0] = x_random
    X[i, 1] = y_random

    if x_random > y_random:
        T[i] = 1

eta = 0.01  # 学习率设定


# --- 逻辑回归，sigmoid函数进行分类 ---
def data_classify(x, a_params, b_param):
    u = np.dot(x, a_params) + b_param
    return 1 / (1 + np.exp(-u))


# --- （交叉熵，适合二分类任务）误差函数 ---
def cross_entropy(Y, T):
    delta = 1e-7
    return -np.sum(T * np.log(Y + delta) + (1 - T) * np.log(1 - Y + delta))


# --- 计算每个参数的斜率 ---
def grad_a_params(X, T, a_params, b_param):
    grad_a = np.zeros(len(a_params))
    for i in range(len(a_params)):
        for j in range(len(X)):
            grad_a[i] = grad_a[i] + (data_classify(X[j], a_params, b_param) - T[j]) * X[j, i]
    return grad_a

def grad_b_param(X, T, a_params, b_param):
    grad_b = 0
    for i in range(len(X)):
        grad_b = grad_b + (data_classify(X[i], a_params, b_param) - T[i])
    return grad_b


# --- 学习相关的函数定义 ---
def fit(X, T, dim, epoch):
    # --- 参数初始化 ---
    a_params = np.random.randn(dim)
    b_param = np.random.randn()


    # --- 更新参数的公式 ---
    for i in range(epoch):
        grad_a = grad_a_params(X, T, a_params, b_param)
        grad_b = grad_b_param(X, T, a_params, b_param)
        a_params = a_params - eta * grad_a
        b_param  = b_param - eta * grad_b
    return (a_params, b_param)

# 开始更新参数、进行分类
# 开始学习
a_params, b_param = fit(X, T, 2, 500)
# 使用学习后的参数进行分类
Y = data_classify(X, a_params, b_param)



