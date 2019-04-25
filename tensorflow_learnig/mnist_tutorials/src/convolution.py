#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 使用卷积神经网络机分析MNIST手写图片
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
x = tf.placeholder(tf.float32, [None, 28, 28, 1])  # None表示图片的数目
t = tf.placeholder(tf.float32, [None, 10])
# 学习速率
lr = tf.placeholder(tf.float32)
# dropout参数
pkeep = tf.placeholder(tf.float32)

# 查看mnist图片数据：
# mnist.train.images.shape is (55000, 784)
# mnist.train.labels.shape is (55000, )
# img = mnist.train.images[0].reshape(28,28)
# plt.imshow(img)
# plt.show()

#### 定义变量 ####
# 定义卷积核权重w和偏置b
ch1 = 6
ch2 = 12
ch3 = 24
ch4 = 200

w1 = tf.Variable(tf.truncated_normal([6, 6, 1, ch1], stddev=0.1))
b1 = tf.Variable(tf.ones([ch1])/10)

w2 = tf.Variable(tf.truncated_normal([5, 5, ch1, ch2], stddev=0.1))
b2 = tf.Variable(tf.ones([ch2])/10)

w3 = tf.Variable(tf.truncated_normal([4, 4, ch2, ch3], stddev=0.1))
b3 = tf.Variable(tf.ones([ch3])/10)

w4 = tf.Variable(tf.truncated_normal([7*7*ch3, ch4], stddev=0.1))
b4 = tf.Variable(tf.ones([ch4])/10)

w5 = tf.Variable(tf.truncated_normal([ch4, 10]))
b5 = tf.Variable(tf.ones([10])/10)

#### 定义模型 ####
# 定义卷积网络，使用relu加快收敛速度，且能防止梯度消失。
# input layer, input is [None, 28, 28, 1]
y1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
# y1 = tf.nn.dropout(y1, rate=1-pkeep)

# input is [None, 28, 28, ch1]
y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, strides=[
                1, 2, 2, 1], padding='SAME') + b2)
# y2 = tf.nn.dropout(y2, rate=1-pkeep)

# input is [None, 14, 14, ch2]
y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, strides=[
                1, 2, 2, 1], padding='SAME') + b3)
# y3 = tf.nn.dropout(y3, rate=1-pkeep)

# flatten, input is [None, 7, 7, ch3]
y3f = tf.reshape(y3, [-1, 7*7*ch3])
y4 = tf.nn.relu(tf.matmul(y3f, w4)+b4)
y4d = tf.nn.dropout(y4, rate=1-pkeep)

# output layer, input is [None, ch4], ouput is [None, 10]
logits = tf.matmul(y4d, w5) + b5
y = tf.nn.softmax(logits)

#### 损失函数 ####
# cross_entropy = -tf.reduce_sum(t * tf.log(y))
# 使用输出层线性加权和作为交叉熵的输入
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=t)

tf.summary.scalar('cross_entropy', 1)

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
        batch_xs_img = batch_xs.reshape([-1, 28, 28, 1])

        # batch_ys = tf.one_hot(batch_ys,10)
        batch_ys_one_hot = to_categorical(batch_ys, 10)
        sess.run(train_step, feed_dict={
                 x: batch_xs_img, t: batch_ys_one_hot, pkeep: 0.75, lr: lr_decay(epoch)})        

        if epoch % 100 == 0:
            acc, loss= sess.run([accuracy, cross_entropy], feed_dict={
                                 x: batch_xs_img, t: batch_ys_one_hot, pkeep: 1})
            labels_one_hot = to_categorical(mnist.test.labels, 10)
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={
                                 x: mnist.test.images.reshape([-1, 28, 28, 1]), t: labels_one_hot, pkeep: 1})
            print("The accaracy is %.5f, and the loss is %.5f." %
                  (acc, sum(loss)))  # The accaracy is 0.97600, and the loss is 715.25799.#