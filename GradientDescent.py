# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 10:55:10 2019

@author: cash_
"""

import tensorflow as tf;
import numpy as np
learning_rate = 0.01
n_epochs = 1000
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)),housing.data]
x = tf.constant(housing_data_plus_bias,dtype=tf.float32,name = "x")
y = tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name = "y")
theta = tf.Variable(tf.random_uniform([n+1,1],-1,1),name = "theta")
y_pred = tf.matmul(x,theta,name = "predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name="mse")
gradients = 2/m*tf.matmul(tf.transpose(x),error)
training_op = tf.assign(theta,theta - learning_rate*gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 ==0:
            print("Epoch",epoch,"MSE",mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    
print(best_theta)
    
        

