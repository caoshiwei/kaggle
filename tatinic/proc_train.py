#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/30 16:14
# @Author  : Cao Shiwei
# @File    : proc_feature_data.py


import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.model_selection import cross_val_score

import re


cabins_dict = dict()
name_dict = dict()


def load_dict(ori_data):
    global cabins_dict
    cabins_set = set()
    for item in list(ori_data['Cabin']):
        #cabins = str(item).split(sep=' ')
        cabins_set.add(item)
    # print("set:")
    # print(cabins_set)
    cabins_dict = dict(zip(cabins_set, range(len(cabins_set))))
    # print(cabins_dict)

    global name_dict
    name_set = set()
    for item in list(ori_data['Name']):
        # print(item)
        m_obj = re.match(r'(.*), (.*\.) *', item)
        if m_obj:

            title = m_obj.group(2)
            name_set.add(title)
            # print(first_name)
            # print(title)

    name_dict = dict(zip(name_set, range(len(name_set))))
        # cabins_set.add(item)


def proc_data(ori_data):
    cabin_nu = list()
    for item in list(ori_data['Cabin']):
        if item in cabins_dict:
            cabin_nu.append(cabins_dict[item])
        else:
            cabin_nu.append(-1)

    name_title = list()
    first_name_len = list()
    for item in list(ori_data['Name']):
        # print(item)
        m_obj = re.match(r'(.*), (.*\.) *', item)
        if m_obj:
            first_name = m_obj.group(1)
            first_name_len.append(len(first_name))
            title = m_obj.group(2)
            if title in name_dict:
                name_title.append(name_dict[title])
            else:
                name_title.append(-1)
        else:
            name_title.append(-1)
            first_name_len.append(0)

    ori_data.loc[(ori_data.Cabin.notnull()), 'Cabin'] = 1
    ori_data.loc[(ori_data.Cabin.isnull()), 'Cabin'] = 0

    ori_data.loc[ori_data['Sex'] == 'male', 'Sex'] = 1
    ori_data.loc[ori_data['Sex'] == 'female', 'Sex'] = 0

    ori_data.loc[ori_data['Embarked'] == 'S', 'Embarked'] = 0
    ori_data.loc[ori_data['Embarked'] == 'C', 'Embarked'] = 1
    ori_data.loc[ori_data['Embarked'] == 'Q', 'Embarked'] = 2
    ori_data.loc[(ori_data.Embarked.isnull()), 'Embarked'] = -1

    ori_data.loc[(ori_data.Fare.isnull()), 'Fare'] = -1

    family = ori_data['Parch'] + ori_data['SibSp']
    ori_data.loc[(ori_data.Age.isnull()), 'Age'] = int(ori_data.Age.mean())
    ori_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    # print(ori_data[0:5])
    ori_data['Cabin_nu'] = cabin_nu
    ori_data['family'] = family
    ori_data['Name_title'] = name_title
    ori_data['Name_len'] = first_name_len
    #print(ori_data[0:5])
    return np.array(ori_data, dtype=np.float32)


def train(train_data, test_data):
    train_y = train_data[:, 0]
    train_x = train_data[:, 1:]

    gbdt_model = GBDT(learning_rate=0.06,
                      n_estimators=300,
                      max_leaf_nodes=16,
                      min_samples_split=2,
                      max_depth=6)
    print(cross_val_score(gbdt_model, train_x, train_y, cv=5))
    gbdt_model.fit(train_x, train_y)
    print(gbdt_model.feature_importances_)

    test_y = gbdt_model.predict(test_data)

    ori_test_data = pd.read_csv("./data/test.csv")
    result = pd.DataFrame({'PassengerId': ori_test_data['PassengerId'].as_matrix(), 'Survived': test_y.astype(np.int32)})
    # print(result)
    result.to_csv("./data/result_test.csv", index=False)


if __name__ == "__main__":
    ori_train_data = pd.read_csv("./data/train.csv")
    load_dict(ori_train_data)

    ori_test_data = pd.read_csv("./data/test.csv")
    data_train = proc_data(ori_train_data)
    data_test = proc_data(ori_test_data)
    # print(data_train[0:5])
    # print(test_data)
    train(data_train, data_test)

    # ori_train_data.info()

