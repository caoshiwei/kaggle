#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/2 11:48
# @Author  : Cao Shiwei
# @File    : train.py

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score


def read_data(filenm):
    pd_data = pd.read_csv(filenm)
    return np.array(pd_data, dtype=np.float32)


def proc_feature(data):
    data = np.array(data) / 255
    return data


def train(X, y):
    svm_c = svm.SVC(kernel='rbf', decision_function_shape='ovo')
    print("svm_cross score ")
    print(cross_val_score(svm_c, X, y, cv=2, scoring='accuracy', n_jobs=-1))
    svm_c.fit(X, y)
    return svm_c


if __name__ == '__main__':
    train_matrix = read_data("./data/train.csv")
    test_matrix = read_data("./data/test.csv")

    train_X = train_matrix[:, 1:]
    train_y = train_matrix[:, 0]
    print(train_X[0:2, 200:300])

    train_X = proc_feature(train_X)
    print("after proc")
    print(train_X[0:2, 200:300])

    model = train(train_X, train_y)

    pred_x = proc_feature(test_matrix)
    pred_y = model.predict(pred_x)
    result = pd.DataFrame({'ImageId': range(1, len(pred_y) + 1),
                           'Label': pred_y.astype(np.int32)})
    # print(result)
    result.to_csv("./data/result_test.csv", index=False)
    print("over")