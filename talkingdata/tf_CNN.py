#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/5 18:52
# @Author  : Cao Shiwei
# @File    : tf_CNN.py

import os

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

import random


class Model():
    def __init__(self, args, training):
        self.args = args
        self.training = not(args.test)
        layers_n = [128, 128]
        self.X = tf.placeholder(tf.float32, [None, 5], name='x_inputs')
        self.label = tf.placeholder(tf.float32, [None, 2], name='label')

        [x_mean, x_varia] = tf.nn.moments(self.X, axes=0)
        offset = 0
        scale = 0.1
        vari_epsl = 0.0001
        X_bn = tf.nn.batch_normalization(self.X, x_mean, x_varia, offset, scale, vari_epsl)
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        layers = []
        for i, n_layer in enumerate(layers_n):
            if i is 0:
                hidden_ = tf.layers.dense(X_bn, units=n_layer, activation=tf.nn.relu, kernel_regularizer=regularizer)
            else:
                hidden_ = tf.layers.dense(layers[-1], units=n_layer, activation=tf.nn.relu, kernel_regularizer=regularizer)
            layers.append(hidden_)

        print(layers[-1].get_shape().as_list())

        #dropout = tf.layers.dropout(layers[-1], rate=0.9, training=True, name='d_dense0')
        logits = tf.layers.dense(layers[-1], 2, name='dense_logits')
        # dropout = tf.nn.dropout(layers[-1], keep_prob=0.65)
        self.prob_logits = tf.nn.softmax(logits)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label))
        self.auc = tf.metrics.auc(self.label, self.prob_logits)

    def train(self):
        opt_ = tf.train.AdadeltaOptimizer(self.args.lr).minimize(self.loss)

        with tf.Session() as sess:
            # sess.run(init)
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            saver = tf.train.Saver(tf.global_variables())

            train_x, labels = read_data('./data/train.csv', train=True)
            print(labels.shape)
            datasize = train_x.shape[0]
            print(datasize)
            print("itrs: %d" % self.args.itrs)
            # valid data
            test_data_size = 10000
            ids = random.sample(range(datasize), test_data_size)
            batch_test_x = train_x[ids]
            batch_test_labels = labels[ids]
            print(batch_test_labels.shape)

            for iter in range(self.args.itrs):
                ids = random.sample(range(datasize), self.args.batch_size)
                batch_x = train_x[ids]
                batch_labels = labels[ids]
                #print(batch_labels.shape)
                loss_value, _ = sess.run([self.loss, opt_], feed_dict={self.X: batch_x, self.label: batch_labels})

                if iter % 200 == 0:
                    print('--------200 steps-------')

                    out = sess.run(self.prob_logits, feed_dict={self.X: batch_test_x})
                    test_auc = sess.run(self.auc, feed_dict={self.X: batch_test_x, self.label: batch_test_labels})
                    out_labels_idx = np.argmax(out, axis=1)
                    labels_idx = np.argmax(batch_test_labels, axis=1)
                    actual = len([i for i in range(test_data_size) if labels_idx[i] == out_labels_idx[i]]) / np.float32(
                        test_data_size)
                    print('itr: {}:loss {}:actual {}'.format(iter, loss_value,
                                                             actual))  # out_label = np.argmax(out, axis=1)
                    print("test_auc: {}".format(test_auc))

                if iter % 2000 == 0:
                    print("After %d training steps, loss on training batch is %g" % (iter, loss_value))
                    print('------------------save model---------------')
                    saver.save(sess, './save/model.ckpt')

    def test(self, test_file, save_as_csv=True):
        # logits_array = []
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, './save/model.ckpt')

            test_data = read_data(test_file, train=False)
            logits_array = sess.run(self.prob_logits, feed_dict={self.X: test_data})
            return logits_array

        #indexs = [np.argmax(logits, axis=1) for logits in logits_array]
        #return indexs


def read_data(filenm, train=True):
    if train:
        df = pd.read_csv(filenm, nrows=1000000)
        is_attri = np.array(df['is_attributed'], dtype=np.int32)
        size = is_attri.shape[0]
        label = np.zeros((size,2))
        label[np.arange(size), is_attri] = 1
        df.drop(['click_time', 'attributed_time', 'is_attributed'], axis=1, inplace=True)
        train_data = np.array(df, dtype=np.int32)
        return train_data, label
    else:
        df = pd.read_csv(filenm)
        df_test = df.drop(['click_time', 'click_id'], axis=1, inplace=False)
        test_data = np.array(df_test, dtype=np.int32)
        return test_data


if __name__ == '__main__':
    argp = argparse.ArgumentParser(description='')
    argp.add_argument('--lr', dest='lr', type=float, default=0.001)
    argp.add_argument('--batch_size', dest='batch_size', type=int, default=128)
    argp.add_argument('--itrs', dest='itrs', type=int, default=19001)
    argp.add_argument('--test', dest='test', action='store_true')
    argp.add_argument('--no-test', dest='test', action='store_false')
    argp.set_defaults(test=False)
    args = argp.parse_args()

    if not os.path.exists('./save/'):
        os.mkdir('./save/')

    if not args.test:
        print("def model")
        model_ = Model(args, True)
        print("begin train")
        model_.train()
        #del model_
        #model_ = Model(args, False)
        print("begin test")
        prob_res = model_.test('./data/test.csv')

        df_test_ori = pd.read_csv('./data/test.csv')
        print(prob_res.shape)
        print(prob_res[:3])
        size = prob_res.shape[0]
        result = pd.DataFrame(
            {'click_id': df_test_ori['click_id'], 'is_attributed': prob_res[:,1].reshape(size).astype(np.float32)})
        print(result.head(10))
        result.to_csv("./data/result_test.csv", index=False)
    else:
        print("test def model")
        model_ = Model(args, False)
        prob_res = model_.test('./data/test.csv')

        df_test_ori = pd.read_csv('./data/test.csv')
        print(prob_res.shape)
        print(prob_res[:3])
        size = prob_res.shape[0]
        result = pd.DataFrame(
            {'click_id': df_test_ori['click_id'], 'is_attributed': prob_res[:,1].reshape(size).astype(np.float32)})
        print(result.head(10))
        result.to_csv("./data/result_test.csv", index=False)
