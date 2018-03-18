#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/8 15:28
# @Author  : Cao Shiwei
# @File    : analy_data.py


import pandas as pd


train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

train_data.info()
print(train_data.describe())

test_data.info()
print(test_data.describe())