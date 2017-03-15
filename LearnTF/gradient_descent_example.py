#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


# x_data, y_data are ndarray in Numpy
x_data = np.random.rand(10,1000)  # uniform distribution
y_data = x_data * 0.2 + 0.5

# W, b are varible
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
# y, loss are tensor, y's name is Add:0, loss' name is Mean:0
y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data))

model = tf.train.GradientDescentOptimizer(0.5)
# train, init are an operation
train = model.minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print("1. (Step:%d)"%step, sess.run(W), sess.run(b))
        print("2. (Step:%d)"%step, W.eval(sess), b.eval(sess)) # same as last
