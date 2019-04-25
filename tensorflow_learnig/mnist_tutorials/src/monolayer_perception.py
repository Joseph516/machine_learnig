#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 使用单层感知机逻辑回归分析MNIST手写图片
@Author: Joe
@Verdion: 
'''
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#### 数据输入 ####
# 导入mnist数据
mnist = input_data.read_data_sets(
    "../data/MNIST_data/", reshape=True, one_hot=False)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
t = tf.placeholder(tf.float32, [None, 10])

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
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 初始化变量
init_op = tf.global_variables_initializer()

#### 定义模型 ####
# 定义单层感知机
y = tf.nn.softmax(tf.matmul(x, w) + b)

#### 损失函数 ####
cross_entropy = -tf.reduce_sum(t * tf.log(y))

# y,t中的最大值对应预测量
is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#### 优化函数 ####
# 设置梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)

train_step = optimizer.minimize(cross_entropy)

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
        sess.run(train_step, feed_dict={x: batch_xs, t: batch_ys_one_hot})

        if epoch % 100 == 0:
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={
                                 x: batch_xs, t: batch_ys_one_hot})
            labels_one_hot = to_categorical(mnist.test.labels, 10)
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={
                                 x: mnist.test.images, t: labels_one_hot})
            print("The accaracy is %.5f, and the loss is %.5f." % (acc, loss))


