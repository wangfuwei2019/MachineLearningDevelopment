# TensorFlow版 (MNIST-手写数字识别)
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics


# 创建类
class DeepNeuralNetwork(Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        # 无需手动添加输入层维度信息
        self.layer01 = Dense(hidden_dim, activation='sigmoid')  # 添加中间层
        self.layer02 = Dense(hidden_dim, activation='sigmoid')  # 添加中间层
        self.layer03 = Dense(output_dim, activation='softmax')  # 添加输出层

        self.all_layers = [self.layer01, self.layer02, self.layer03]  # 需要手动将各层组织起来

    # 前向传播以前是forward函数，现在是call函数，应引起注意
    def call(self, x):
        for layer in self.lys:
            x = layer(x)
        return x


if __name__ == '__main__':
    np.random.seed(10)
    tf.random.set_seed(10)

    # 1. 准备MNIST数据
    mnist = datasets.mnist
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    # 将三维的数据转化为二维的数据，更有利于神经网络的传输
    # 将像素值范围（0-255）转化为值范围（0-1）之间
    x_train = (x_train.reshape(-1, 784) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 784) / 255).astype(np.float32)
    # np.eye()函数实现了独热编码
    t_train = np.eye(10)[t_train].astype(np.float32)
    t_test = np.eye(10)[t_test].astype(np.float32)

    #  2. 建模
    model = DeepNeuralNetwork(200, 10)

    #  3. 学习
    # 调用keras API 快速建立适合多分类任务的交叉熵对象
    criterion = losses.CategoricalCrossentropy()
    # 调用keras API 快速建立优化器对象，并指定学习率
    optimizer = optimizers.SGD(learning_rate=0.01)

    train_loss = metrics.Mean()  # 提供获取训练阶段误差值的对象
    train_accuracy = metrics.CategoricalAccuracy()  # 提供训练阶段精确度的对象


    # 将误差值封装为函数
    def compute_loss(t, y):
        return criterion(t, y)


    # 梯度计算、误差值、精确度的封装函数
    def train_step(x, t):
        with tf.GradientTape() as tape:
            predict_value = model(x)
            loss = compute_loss(t, predict_value)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss(loss)
        train_accuracy(t, predict_value)

        return loss


    epochs = 30  # 更新参数的总轮数
    batch_size = 100  # 批处理的大小
    # 将样本一共划分为多少份
    n_batches = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        x_, t_ = shuffle(x_train, t_train)  # 将学习数据打乱顺序，防止局部最小值、过\欠拟合现象的发生

        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            train_step(x_[start:end], t_[start:end])

        print('The epoch is: {}, The loss is: {:.3}, The accuracy is: {:.3f}'.format(
            epoch + 1, train_loss.result(), train_accuracy.result())
        )
