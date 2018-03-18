#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 11:46
# @Author  : Cao Shiwei
# @File    : talking_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as GBDT_C
from sklearn.linear_model import SGDClassifier as SGD_C
from sklearn.model_selection import cross_val_score


def iter_minibatches(train_file, minibatch_size=1024*100):
    #with open(train_file) as f:
        #df_train = pd.read_csv(f, nrows=minibatch_size)
    for df_train in pd.read_csv(train_file, chunksize=minibatch_size):
        print('Size of uploaded chunk: %i instances, %i features' % (df_train.shape))
        #print(df_train)
        label = np.array(df_train['is_attributed'], dtype=np.int32)
        df_train.drop(['click_time', 'attributed_time', 'is_attributed'], axis=1, inplace=True)
        train_X = np.array(df_train, dtype=np.int32)
        yield train_X, label


def train(train_file):
    df_train = pd.read_csv(train_file, nrows=100000)
    label = np.array(df_train['is_attributed'], dtype=np.int32)
    df_train.drop(['click_time', 'attributed_time', 'is_attributed'], axis=1, inplace=True)
    print(df_train.head(10))
    train_data = np.array(df_train, dtype=np.int32)
    gbdt_model = GBDT_C(learning_rate=0.05,
                        n_estimators=250,
                        max_leaf_nodes=8,
                        min_samples_split=6,
                        max_depth=3)
    print("GBDT cross score:")
    print(cross_val_score(gbdt_model, train_data, label, cv=3, scoring='accuracy'))

    gbdt_model.fit(X=train_data, y=label)
    return gbdt_model


def test(model, test_file):
    df_test_ori = pd.read_csv(test_file)
    df_test = df_test_ori.drop(['click_time', 'click_id'], axis=1, inplace=False)
    test_data = np.array(df_test, dtype=np.int32)
    y_ = model.predict_proba(test_data)
    print(y_[:3])
    size = (y_.shape)[0]
    result = pd.DataFrame({'click_id': df_test_ori['click_id'],
                           'is_attributed': y_[:1].reshape(size).astype(np.float32)})
    print(result.head(10))
    result.to_csv("./data/result_test.csv", index=False)


def batch_train(train_file):
    minibatch_train_iterators = iter_minibatches(train_file, minibatch_size=1024*100)
    X_test, y_test = next(minibatch_train_iterators)  # 得到一份测试文件
    model_SGD = SGD_C(loss='log')
    for i, (X_train, y_train) in enumerate(minibatch_train_iterators):
        # 使用 partial_fit ，并在第一次调用 partial_fit 的时候指定 classes
        model_SGD.partial_fit(X_train, y_train, classes=[0, 1])
        print("{} time".format(i))  # 当前次数
        print("{} score".format(model_SGD.score(X_test, y_test)))  # 在测试集上看效果

    return model_SGD


if __name__ == '__main__':
    train_file = "./data/train.csv"
    test_file = "./data/test.csv"
    model = train(train_file)
    #model = batch_train(train_file)
    print("Begin predict:")
    test(model, test_file)
