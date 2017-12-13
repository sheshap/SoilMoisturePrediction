# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 17:16:22 2017

@author: Manisha
"""

import numpy as np
import math
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import tensorflow as tf
import csv

## CNN parameters

segment_size = 5
num_input_channels = 3

num_training_iterations = 50
batch_size = 100

l2_reg = 5e-4
learning_rate = 5e-4
dropout_rate = 0.05
eval_iter = 1000

n_filters = 196
filters_size = 16
n_hidden = 1024
n_classes = 3


def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv1d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1x4(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='SAME')


def norm(x):
    temp = x.T - np.mean(x.T, axis=0)
    # temp = temp / np.std(temp, axis = 0)
    return temp.T


## Loading the dataset

print('Loading UCI dataset...')

# Reading training data

fa = open("C:\\Users\\makri\\Desktop\\New Folder\\Site674_mice_20inchSMS.csv")
ff = open("C:\\Users\\makri\\Desktop\\New Folder\\Site674_mice_pca_20inchSMS.csv")

data_train = np.loadtxt(fname=fa, delimiter=',')
features = np.loadtxt(fname=ff, delimiter=',')

fa.close();
ff.close()

# Reading training labels

fa = open("C:\\Users\\makri\\Desktop\\New Folder\\Site674_mice_20inchSMS_value.csv")
labels_train = np.loadtxt(fname=fa, delimiter=',')
fa.close()


features = features - np.mean(features, axis=0)
features = features / np.std(features, axis=0)



for i in range(num_input_channels):
    x = data_train[:, i * segment_size: (i + 1) * segment_size]
    data_train[:, i * segment_size: (i + 1) * segment_size] = norm(x)
 

train_size = data_train.shape[0]

num_features = features.shape[1]



print("Dataset was uploaded\n")

## creating CNN

print("Creating CNN architecture\n")

# Convolutional and Pooling layers

W_conv1 = weight_variable([1, filters_size, num_input_channels, n_filters], stddev=0.01)
b_conv1 = bias_variable([n_filters])

x = tf.placeholder(tf.float32, [None, segment_size * num_input_channels])
x_image = tf.reshape(x, [-1, 1, segment_size, num_input_channels])

h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1) + b_conv1)
h_conv1=tf.layers.conv2d(x_image, 196, 16,strides=(1,1), padding="same", activation=tf.nn.relu)

h_pool1 = max_pool_1x4(h_conv1)


# Augmenting data with statistical features

flat_size = int(math.ceil(float(segment_size) / 4)) * n_filters

h_feat = tf.placeholder(tf.float32, [None, num_features])
h_flat = tf.reshape(h_pool1, [-1, flat_size])

h_hidden = tf.concat([h_flat, h_feat],1)

h_hidden = tf.Print(h_hidden, [h_hidden])
flat_size += num_features

W_fc1 = weight_variable([flat_size, n_hidden], stddev=0.01)
b_fc1 = bias_variable([n_hidden])

h_fc1 = tf.nn.relu(tf.matmul(h_hidden, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax layer

W_softmax = weight_variable([n_hidden, n_classes], stddev=0.01)
b_softmax = bias_variable([n_classes])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_softmax) + b_softmax)
y_ = tf.placeholder(tf.float32, [None, n_classes])

# Cross entropy loss function and L2 regularization term

cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
cross_entropy += l2_reg * (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1))

# Training step

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run Tensorflow session

# Run Tensorflow session

sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Train CNN
print("Training CNN... ")

max_accuracy = 0.0


        
for i in range(num_training_iterations):

    idx_train = np.random.randint(0, train_size, batch_size)

    xt = np.reshape(data_train[idx_train], [data_train[idx_train].shape[0], data_train[idx_train].shape[1]])
    yt = np.reshape(labels_train[idx_train], [batch_size, n_classes])
    ft = np.reshape(features[idx_train], [batch_size, num_features])
    #xt=(data_train[idx_train])
    #yt=(labels_train[idx_train])
    #ft = (features[idx_train])

    sess.run(train_step, feed_dict={x: xt, y_: yt, h_feat: ft, keep_prob: dropout_rate})
    pool = sess.run(h_pool1,feed_dict={x: xt, y_: yt, h_feat: ft, keep_prob: dropout_rate})
    with open('C:\\Users\\makri\\Desktop\\features.csv', 'a') as csvfile:
           spamwriter = csv.writer(csvfile, delimiter=',')
           spamwriter.writerow(pool[0][0][0][:7])
        
    
