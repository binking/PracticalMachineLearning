# -*- codng: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load data set
mnist_data = input_data.read_data_sets('../MNIST_data', one_hot=True)

# build graph
x = tf.placeholder(tf.float32, [None, 784])  # size: 1 * 784
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
predict = tf.nn.softmax(tf.matmul(x, W) + b)
label = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(- tf.reduce_sum(label * tf.log(predict), reduction_indices=[1])) # our loss fucntion
model = tf.train.GradientDescentOptimizer(0.5)  # 0.5 is learning rate
train_step = model.minimize(cross_entropy)  # target

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(100):  # train 1000 times
    batch_x, batch_y = mnist_data.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_x, label: batch_y})
