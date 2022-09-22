import numpy as np
# -- 准备数据 --
input_data = np.arange(0, np.pi * 2, 0.1)  # 输入数据
t_data = np.sin(input_data)  # 对应的正确答案，也就是函数值，范围在[0,1]
input_data = (input_data - np.pi) / np.pi  # 将输入数据控制在[-1.0-1.0]之间
n_data = len(t_data)  # 求样本的数量

# -- 网络中各神经元维度和相关参数初始值的设定 --
n_input = 1  # 输入层的神经元数量
n_mid = 10  # 中间层的神经元数量
n_output = 1  # 输出层的神经元数量

eta = 0.1  # 学习系数
epoch = 16000

# -- 中间层网络的设计与实现 --
class MiddleLayer:
    def __init__(self, n_upper, n):  # 参数初始化
        self.w = np.random.randn(n_upper, n)  # 权重
        self.b = np.random.randn(n)  # 偏置

    def forward(self, x):  # 正向传播
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = 1 / (1 + np.exp(-u))  # Sigmoid函数

    def backward(self, grad_y):  # 反向传播
        conclusion = grad_y * (1 - self.y) * self.y  # Sigmoid函数求导后的公式

        self.grad_w = np.dot(self.x.T, conclusion)
        self.grad_b = np.sum(conclusion, axis=0)

        self.grad_x = np.dot(conclusion, self.w.T)

    def update(self, eta):  # 权重和偏置的更新
        self.w = self.w - eta * self.grad_w
        self.b = self.b - eta * self.grad_b


# -- 输出层网络的设计与实现 --
class OutputLayer:
    def __init__(self, n_upper, n):  # 参数初始化
        self.w = np.random.randn(n_upper, n)  # 权重
        self.b = np.random.randn(n)  # 偏置

    def forward(self, x):  # 正向传播
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = u  # 恒等函数，这点与中间层注意区别

    def backward(self, t):  # 反向传播
        conclusion = self.y - t

        self.grad_w = np.dot(self.x.T, conclusion)
        self.grad_b = np.sum(conclusion, axis=0)

        self.grad_x = np.dot(conclusion, self.w.T)

    def update(self, eta):  # 权重和偏置的更新
        self.w = self.w - eta * self.grad_w
        self.b = self.b - eta * self.grad_b


# -- 反向传播的实施 --
middle_layer = MiddleLayer(n_input, n_mid)
output_layer = OutputLayer(n_mid, n_output)

# -- 开始学习 --
for i in range(epoch):

    index_random = np.arange(n_data)
    np.random.shuffle(index_random)

    for idom in index_random:
        x = input_data[idom:idom + 1]  # 输入
        t = t_data[idom:idom + 1]  # 正确答案
        # 正向传播
        middle_layer.forward(x.reshape(1, 1))
        output_layer.forward(middle_layer.y)
        # 反向传播
        output_layer.backward(t.reshape(1, 1))
        middle_layer.backward(output_layer.grad_x)

        # 权重和偏置的更新
        middle_layer.update(eta)
        output_layer.update(eta)
