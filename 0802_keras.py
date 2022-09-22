# Keras版 (MNIST手写数字识别)

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

if __name__ == '__main__':
    np.random.seed(10)
    tf.random.set_seed(10)

    # 1. 准备数据
    mnist = datasets.mnist
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    # 将三维的数据转化为二维的数据，更有利于神经网络的传输
    # 将像素值范围（0-255）转化为值范围（0-1）之间
    x_train = (x_train.reshape(-1, 784) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 784) / 255).astype(np.float32)
    # np.eye()函数实现了独热编码
    t_train = np.eye(10)[t_train].astype(np.float32)
    t_test = np.eye(10)[t_test].astype(np.float32)

    # 2. 建模
    model = Sequential()
    model.add(Dense(200, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # 3. 学习
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, t_train, epochs=30, batch_size=100, verbose=2)
