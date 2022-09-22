import numpy as np
import matplotlib.pyplot as plt

# 数据准备
X = np.linspace(0, np.pi)
T = (np.cos(X) + 1)/2

# --- 单一神经元的正向传播 sigmoid函数---
def forward(x, w, b):
    y = 1/(1+np.exp(-(x*w + b)))
    return y

# --- 单一神经元的反向传播 ---
def backward(x, y, t):
    grad_w = x * ((y - t)*(1-y)*y)
    grad_b = (y - t)*(1-y)*y
    return (grad_w, grad_b)

# 可视化
def output(X, Y, T):
    plt.plot(X, T, linestyle="dashdot", color="red")
    plt.scatter(X, Y, color="green", marker=".", linestyle="solid")

    plt.xlabel("x", size=15)
    plt.ylabel("y", size=15)
    plt.grid()
    plt.show()

# --- 初始值设定 ---
eta = 0.02  # 学习率
epoch = 160  # 完成一轮所有数据的学习的次数

# --- 初始值的设定 ---
w = 0.2  # 权重
b = -0.2  # 偏置

# --- 学习 ---
for i in range(epoch):
    integer_numbers = np.arange(50)
    np.random.shuffle(integer_numbers)

    for j in integer_numbers:

        x = X[j]
        t = T[j]

        y = forward(x, w, b)
        grad_w, grad_b = backward(x, y, t)
        w = w - eta * grad_w
        b = b - eta * grad_b

# --- 最终结果 ---
Y = forward(X, w, b)
output(X, Y, T)



