#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/5 18:52
# @Author  : Cao Shiwei
# @File    : tf_CNN.py

import numpy as np
import pandas as pd
import tensorflow as tf

import random


def read_data(filenm, train=True):
    pf = pd.read_csv(filenm)
    ori_data = np.array(pf, dtype=np.float32)
    imgs = []
    labels = []
    for line in ori_data:
        if train:
            imgs.append(line[1:].reshape(28,28,1))
            label = np.zeros(10)
            label[int(line[0])] = 1
            labels.append(label)
        else:
            imgs.append(line.reshape(28, 28, 1))
    if train:
        return np.array(imgs, dtype=np.float32) / 255.0, np.array(labels, dtype=np.float32)
    else:
        return np.array(imgs, dtype=np.float32) / 255.0


def train(steps_num, batch_size):
    filters = [32, 64]
    X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x_inputs')
    label = tf.placeholder(tf.float32, [None, 10], name='label')
    conv_layer = []
    for i, filter in enumerate(filters):
        if i is 0:
            conv_ = tf.layers.conv2d(X, filter,
                                     [5,5], padding='SAME',
                                     activation=tf.nn.relu,
                                     name='conv_{}_0'.format(i))
            conv_layer.append(conv_)
        else:
            conv_ = tf.layers.conv2d(conv_layer[-1], filter,
                                     [5,5], padding='SAME',
                                     activation=tf.nn.relu,
                                     name='conv_{}_0'.format(i))
            conv_layer.append(conv_)
        pool = tf.layers.max_pooling2d(conv_, [2,2], strides=(2,2))
        conv_layer.append(pool)
    print(conv_layer[-1].get_shape().as_list())
    flatten = tf.reshape(conv_layer[-1], (-1, 7*7*64))
    dense0 = tf.layers.dense(flatten, 512, tf.nn.relu, name='dense0')
    dropout = tf.layers.dropout(dense0, rate=0.65, training=True, name='d_dense0')
    logits = tf.layers.dense(dropout, 10, name='dense_logits')
    prob_logits = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))

    opt_ = tf.train.AdadeltaOptimizer().minimize(loss)
    #config = tf.ConfigProto()
    #config.log_device_placement = True
    # batch_x, batch_labels = tf.train.shuffle_batch([train_x, labels], batch_size=batch_size, num_threads=4,
    #                                                capacity=50000, min_after_dequeue=10000)

    with tf.Session() as sess:
        #sess.run(init)
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables())

        # threads = tf.train.start_queue_runners(sess=sess)
        # val, l = sess.run([batch_x, batch_labels])
        train_x, labels = read_data('./data/train.csv', train=True)
        datasize = train_x.shape[0]
        print(datasize)
        for iter in range(steps_num):
            ids = random.sample(range(datasize), batch_size)
            batch_x = train_x[ids]
            batch_labels = labels[ids]
            loss_value, _ = sess.run([loss, opt_], feed_dict={X: batch_x, label: batch_labels})

            if iter % 200 == 0:
                print('--------200 steps-------')
                ids = random.sample(range(datasize), batch_size)
                batch_test_x = train_x[ids]
                batch_test_labels = labels[ids]
                out = sess.run(prob_logits, feed_dict={X: batch_test_x})
                out_labels_idx = np.argmax(out, axis=1)
                labels_idx = np.argmax(batch_test_labels, axis=1)
                actual = len([i for i in range(batch_size) if labels_idx[i] == out_labels_idx[i]]) / np.float32(batch_size)
                print('itr: {}:loss {}:actual {}'.format(iter, loss_value, actual))
                # out_label = np.argmax(out, axis=1)

            if iter % 2000 == 0:
                print("After %d training steps, loss on training batch is %g" % (iter, loss_value))
                print('------------------save model---------------')
                saver.save(sess, './save/model.ckpt')


if __name__ == '__main__':
    steps_num = 10000
    batch_size = 128
    train(steps_num, batch_size)