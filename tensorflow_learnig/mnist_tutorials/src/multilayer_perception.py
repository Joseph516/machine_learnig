#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 使用多层感知机分析MNIST手写图片
@Author: Joe
@Verdion: 
'''
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np


#### 数据输入 ####
# 导入mnist数据
mnist = input_data.read_data_sets(
    "../data/MNIST_data/", reshape=True, one_hot=False)

# 输入输出参数
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
t = tf.placeholder(tf.float32, [None, 10])
# 学习速率
lr = tf.placeholder(tf.float32)
# dropout参数
pkeep = tf.placeholder(tf.float32)

# 扁平化处理，转化成一维向量，28*28-->784
# mnist.train.images.shape is (55000, 784)
# mnist.train.labels.shape is (55000, )
x = tf.reshape(x, [-1, 784])
# t = tf.reshape(t, [-1, 10])

# 查看mnist图片数据：
# img = mnist.train.images[0].reshape(28,28)
# plt.imshow(img)
# plt.show()

#### 定义变量 ####
# 定义权重w和偏置b
# 定义每层感知机数目
K = 200
L = 100
M = 60
N = 30

w1 = tf.Variable(tf.truncated_normal([28*28, K], stddev=0.1))
b1 = tf.Variable(tf.ones([K])/10)

w2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
b2 = tf.Variable(tf.ones([L])/10)

w3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
b3 = tf.Variable(tf.ones([M])/10)

w4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
b4 = tf.Variable(tf.ones([N])/10)

w5 = tf.Variable(tf.truncated_normal([N, 10]))
b5 = tf.Variable(tf.ones([10])/10)

#### 定义模型 ####
# 定义多层感知机，使用relu加快收敛速度，且能防止梯度消失。
# input layer
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
y1d = tf.nn.dropout(y1, rate=1-pkeep)

y2 = tf.nn.relu(tf.matmul(y1d, w2) + b2)
y2d = tf.nn.dropout(y2, rate=1-pkeep)

y3 = tf.nn.relu(tf.matmul(y2d, w3) + b3)
y3d = tf.nn.dropout(y3, rate=1-pkeep)

y4 = tf.nn.relu(tf.matmul(y3d, w4) + b4)
y4d = tf.nn.dropout(y4, rate=1-pkeep)

# output layer
logits = tf.matmul(y4, w5) + b5
y = tf.nn.softmax(logits)

#### 损失函数 ####
# cross_entropy = -tf.reduce_sum(t * tf.log(y))
# 使用输出层线性加权和作为交叉熵的输入
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=t)

# y,t中的最大值对应预测量
is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#### 优化函数 ####
# 设置梯度下降
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_step = optimizer.minimize(cross_entropy)


def lr_decay(step):
    # 学习速率逐渐降低
    max_lr, min_lr, decay_speed = 0.003, 0.0001, 2000.0
    return min_lr + (max_lr-min_lr)*math.exp(-step/decay_speed)

# 初始化变量
init_op = tf.global_variables_initializer()

#### 模型训练 ####
epochs = 1000
with tf.Session() as sess:
    # 初始化
    sess.run(init_op)

    # 开始训练
    for epoch in range(epochs):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # batch_ys = tf.one_hot(batch_ys,10)
        batch_ys_one_hot = to_categorical(batch_ys, 10)
        sess.run(train_step, feed_dict={
                 x: batch_xs, t: batch_ys_one_hot, pkeep: 0.75, lr: lr_decay(epoch)})

        if epoch % 100 == 0:
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={
                                 x: batch_xs, t: batch_ys_one_hot, pkeep: 1})
            labels_one_hot = to_categorical(mnist.test.labels, 10)
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={
                                 x: mnist.test.images, t: labels_one_hot, pkeep: 1})
            print("The accaracy is %.5f, and the loss is %.5f." %
                  (acc, sum(loss)))

# ! 如何可视化acc和loss???
# ! 如何理解feed_dict?