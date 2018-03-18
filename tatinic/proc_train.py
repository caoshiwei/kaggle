#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/30 16:14
# @Author  : Cao Shiwei
# @File    : proc_feature_data.py


import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import scorer
from sklearn.model_selection import GridSearchCV

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


def predict_age(ori_data):
    null_age = ori_data[ori_data.Age.isnull()]
    unnull_age = ori_data[ori_data.Age.notnull()]

    y = np.array(unnull_age['Age'], dtype=np.float32)
    unnull_age.drop(['Age', 'Survived'], axis=1, inplace=True)
    X = np.array(unnull_age, dtype=np.float32)

    #param_test = {'n_estimators': range(100, 1000, 50)}
    rfr = RandomForestRegressor(random_state=0, n_estimators=750, max_depth=16, n_jobs=-1, min_samples_split=16)
    #gsearch = GridSearchCV(estimator=rfr, param_grid=param_test, scoring='neg_mean_absolute_error', cv=3)
    #gsearch.fit(X, y)
    #print(gsearch.best_params_, gsearch.best_score_)

    # print(cross_val_score(rfr, X, y, cv=5, scoring='neg_mean_absolute_error'))
    rfr.fit(X, y)

    null_age.drop(['Age', 'Survived'], axis=1, inplace=True)
    test_x = np.array(null_age, dtype=np.float32)
    predict_y = rfr.predict(test_x)
    ori_data.loc[(ori_data.Age.isnull()), 'Age'] = predict_y
    return rfr


def proc_data(ori_data, test=False):
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
    ori_data.loc[(ori_data.Embarked.isnull()), 'Embarked'] = 2

    ori_data.loc[(ori_data.Fare.isnull()), 'Fare'] = ori_data.Fare.median()

    family = ori_data['Parch'] + ori_data['SibSp']

    ori_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    # print(ori_data[0:5])
    ori_data['Cabin_nu'] = cabin_nu
    ori_data['family'] = family
    ori_data['Name_title'] = name_title
    ori_data['Name_len'] = first_name_len
    ori_data['Fare_bias'] = ori_data['Fare'] - ori_data.Fare.mean()
    ori_data['Fare_level'] = ori_data['Fare'] // 5
    # print(ori_data[0:5])

    if not test:
        global RFR
        RFR = predict_age(ori_data)
    else:
        null_age = ori_data[ori_data.Age.isnull()]

        test_data = null_age.drop(['Age'], axis=1, inplace=False)
        test_x = np.array(test_data, dtype=np.float32)
        predict_y = RFR.predict(test_x)
        ori_data.loc[(ori_data.Age.isnull()), 'Age'] = predict_y

    # ori_data.loc[(ori_data.Age.isnull()), 'Age'] = int(ori_data.Age.median())
    ori_data['Age_level'] = ori_data['Age'] // 10
    ori_data['Age_bias'] = ori_data['Age'] - ori_data.Age.mean()
    return np.array(ori_data, dtype=np.float32)


def norm_data(ori_data, test=False):
    dummies_Embarked = pd.get_dummies(ori_data['Embarked'], prefix='Embarked')
    dummies_Pclass = pd.get_dummies(ori_data['Pclass'], prefix='Pclass')

    #ori_data.loc[(ori_data.Embarked.isnull()), 'Embarked'] = -1

    ori_data.loc[(ori_data.Fare.isnull()), 'Fare'] = ori_data.Fare.median()

    ori_data = pd.concat([ori_data, dummies_Embarked, dummies_Pclass], axis=1)
    ori_data.drop(['Pclass', 'Embarked'], axis=1, inplace=True)

    scaler = preprocessing.StandardScaler()
    if not test:
        global age_scaler
        age_scaler = scaler.fit(ori_data['Age'].values.reshape(-1, 1))
        ori_data['Age'] = scaler.fit_transform(ori_data['Age'].values.reshape(-1,1), age_scaler)

        global fare_scaler
        fare_scaler = scaler.fit(ori_data['Fare'].values.reshape(-1, 1))
        ori_data['Fare'] = scaler.fit_transform(ori_data['Fare'].values.reshape(-1,1), fare_scaler)

        global name_title_scaler
        name_title_scaler = scaler.fit(ori_data['Name_title'].values.reshape(-1, 1))
        ori_data['Name_title'] = scaler.fit_transform(ori_data['Name_title'].values.reshape(-1,1), name_title_scaler)

        global name_len_scaler
        name_len_scaler = scaler.fit(ori_data['Name_len'].values.reshape(-1, 1))
        ori_data['Name_len'] = scaler.fit_transform(ori_data['Name_len'].values.reshape(-1,1), name_len_scaler)

        global age_bias_scaler
        age_bias_scaler = scaler.fit(ori_data['Age_bias'].values.reshape(-1, 1))
        ori_data['Age_bias'] = scaler.fit_transform(ori_data['Age_bias'].values.reshape(-1,1), age_bias_scaler)

        global fare_bias_scaler
        fare_bias_scaler = scaler.fit(ori_data['Fare_bias'].values.reshape(-1, 1))
        ori_data['Fare_bias'] = scaler.fit_transform(ori_data['Fare_bias'].values.reshape(-1,1), fare_bias_scaler)

        global age_level_scaler
        age_level_scaler = scaler.fit(ori_data['Age_level'].values.reshape(-1, 1))
        ori_data['Age_level'] = scaler.fit_transform(ori_data['Age_level'].values.reshape(-1,1), age_level_scaler)

        global fare_level_scaler
        fare_level_scaler = scaler.fit(ori_data['Fare_level'].values.reshape(-1, 1))
        ori_data['Fare_level'] = scaler.fit_transform(ori_data['Fare_level'].values.reshape(-1,1), fare_level_scaler)

        global cabin_nu_scaler
        cabin_nu_scaler = scaler.fit(ori_data['Cabin_nu'].values.reshape(-1, 1))
        ori_data['Cabin_nu'] = scaler.fit_transform(ori_data['Cabin_nu'].values.reshape(-1,1), cabin_nu_scaler)

        global family_scaler
        family_scaler = scaler.fit(ori_data['family'].values.reshape(-1, 1))
        ori_data['family'] = scaler.fit_transform(ori_data['family'].values.reshape(-1,1), family_scaler)

        global parch_scaler
        parch_scaler = scaler.fit(ori_data['Parch'].values.reshape(-1, 1))
        ori_data['Parch'] = scaler.fit_transform(ori_data['Parch'].values.reshape(-1,1), parch_scaler)

        global sibsp_scaler
        sibsp_scaler = scaler.fit(ori_data['SibSp'].values.reshape(-1, 1))
        ori_data['SibSp'] = scaler.fit_transform(ori_data['SibSp'].values.reshape(-1,1), sibsp_scaler)
    else:
        ori_data['Age'] = scaler.fit_transform(ori_data['Age'].values.reshape(-1, 1), age_scaler)
        ori_data['Fare'] = scaler.fit_transform(ori_data['Fare'].values.reshape(-1, 1), fare_scaler)
        ori_data['Name_title'] = scaler.fit_transform(ori_data['Name_title'].values.reshape(-1, 1), name_title_scaler)
        ori_data['Name_len'] = scaler.fit_transform(ori_data['Name_len'].values.reshape(-1, 1), name_len_scaler)
        ori_data['Age_bias'] = scaler.fit_transform(ori_data['Age_bias'].values.reshape(-1, 1), age_bias_scaler)
        ori_data['Fare_bias'] = scaler.fit_transform(ori_data['Fare_bias'].values.reshape(-1, 1), fare_bias_scaler)
        ori_data['Age_level'] = scaler.fit_transform(ori_data['Age_level'].values.reshape(-1, 1), age_level_scaler)
        ori_data['Fare_level'] = scaler.fit_transform(ori_data['Fare_level'].values.reshape(-1, 1), fare_level_scaler)
        ori_data['Cabin_nu'] = scaler.fit_transform(ori_data['Cabin_nu'].values.reshape(-1, 1), cabin_nu_scaler)
        ori_data['family'] = scaler.fit_transform(ori_data['family'].values.reshape(-1, 1), family_scaler)
        ori_data['Parch'] = scaler.fit_transform(ori_data['Parch'].values.reshape(-1, 1), parch_scaler)
        ori_data['SibSp'] = scaler.fit_transform(ori_data['SibSp'].values.reshape(-1, 1), sibsp_scaler)
    print(ori_data[0:5])
    return np.array(ori_data, dtype=np.float32)


def GBDT_train(train_data, test_data):
    train_y = train_data[:, 0]
    train_x = train_data[:, 1:]

    # param_test = {'n_estimators': range(50, 1000, 50)}
    gbdt_model = GBDT(learning_rate=0.05,
                      n_estimators=250,
                      max_leaf_nodes=8,
                      min_samples_split=6,
                      max_depth=3)
    # gsearch = GridSearchCV(estimator=gbdt_model, param_grid=param_test, scoring='accuracy', cv=5)
    # gsearch.fit(train_x, train_y)
    # print(gsearch.best_params_, gsearch.best_score_)

    # bagging_gbdt = BaggingClassifier(gbdt_model, max_samples=0.8)
    print("GBDT cross score:")
    print(cross_val_score(gbdt_model, train_x, train_y, cv=5, scoring='accuracy'))
    #print(cross_val_score(bagging_gbdt, train_x, train_y, cv=5))
    gbdt_model.fit(train_x, train_y)
    test_y = gbdt_model.predict(test_data)
    return test_y


def svm_train(train_data, test_data):
    train_y = train_data[:, 0]
    train_x = train_data[:, 1:]

    # svm
    svm_c = svm.SVC(kernel='rbf', C=4)
    # bagging_svm = BaggingClassifier(svm_c, max_samples=0.8)
    #param_test = {"C": range(1,20,1)}
    # gsearch = GridSearchCV(estimator=svm_c, param_grid=param_test, scoring='accuracy', cv=5)
    # gsearch.fit(train_x, train_y)
    # print(gsearch.best_params_, gsearch.best_score_)
    print("svm score")
    print(cross_val_score(svm_c, train_x, train_y, cv=5))
    #print(cross_val_score(bagging_svm, train_x, train_y, cv=5, scoring='accuracy'))
    svm_c.fit(train_x, train_y)
    test_y = svm_c.predict(test_data)
    return test_y


def forest_train(train_data, test_data):
    train_y = train_data[:, 0]
    train_x = train_data[:, 1:]

    # param_test = {'min_samples_split': range(2, 32, 2)}
    rf_m = RF(
        n_estimators=500,
        max_depth=22,
        min_samples_split=8)
    # gsearch = GridSearchCV(estimator=rf_m, param_grid=param_test, scoring='accuracy', cv=5)
    # gsearch.fit(train_x, train_y)
    # print(gsearch.best_params_, gsearch.best_score_)
    print("RF score")
    print(cross_val_score(rf_m, train_x, train_y, cv=5, scoring='accuracy'))
    rf_m.fit(train_x, train_y)
    test_y = rf_m.predict(test_data)
    return test_y


if __name__ == "__main__":
    ori_train_data = pd.read_csv("./data/train.csv")
    ori_train_data.info()
    print(ori_train_data.describe())
    load_dict(ori_train_data)
    # ori_train_data.info()
    ori_test_data = pd.read_csv("./data/test.csv")

    data_train = proc_data(ori_train_data)
    data_test = proc_data(ori_test_data, test=True)
    test_y1 = forest_train(data_train, data_test)

    test_y2 = GBDT_train(data_train, data_test)

    #ori_train_data = pd.read_csv("./data/train.csv")
    #ori_test_data = pd.read_csv("./data/test.csv")
    data_train = norm_data(ori_train_data)
    data_test = norm_data(ori_test_data, test=True)
    test_y3 = svm_train(data_train, data_test)

    predict_y = (test_y1 + test_y2 + test_y3) // 2
    print(test_y1[0:5])
    print(test_y2[0:5])
    print(test_y3[0:5])
    print(predict_y[0:5])

    ori_test_data = pd.read_csv("./data/test.csv")
    result = pd.DataFrame({'PassengerId': ori_test_data['PassengerId'].as_matrix(),
                           'Survived': predict_y.astype(np.int32)})
    # print(result)
    result.to_csv("./data/result_test.csv", index=False)
