""" 
商品利润最大化
商品成本是1元，利润是10元。多卖1件赚10元，少卖1件少赚10元。
"""
from numpy.random import RandomState
import tensorflow as tf

# 输入数据
dataset_size = 128
rdm = RandomState(1)
X = rdm.rand(dataset_size, 2)
Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1, x2) in X]

# 定义变量
x = tf.placeholder(tf.float32, shape=[None, 2], name='x_input')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y_input')

# 定义模型
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
# b1 = tf.Variable(tf.zeros([1])/10.)
y_pred = tf.matmul(x, w1)

# 自定义loss
loss_more = 10
loss_less = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_pred),
                              loss_more * (y-y_pred), loss_less*(y_pred-y)))
train_step = tf.train.AdamOptimizer().minimize(loss)

# 开始训练
batch_size = 8
epochs = 3000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        start = (epoch*batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run([train_step], feed_dict={x: X[start:end], y: Y[start:end]})

        if (epoch % 500 == 0):
            loss_out = sess.run([loss], feed_dict={x: X, y: Y})
            print("Benefit loss is:",loss_out)
    
    # 输出权重和偏置
    w= sess.run([w1], feed_dict={x: X, y: Y})
    # w, b = sess.run([w1, b1], feed_dict={x: X, y: Y})
    print("Weight is\n:", w)
    # print("Bias is\n:", b)