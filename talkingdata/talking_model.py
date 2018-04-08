#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 11:46
# @Author  : Cao Shiwei
# @File    : talking_model.py

import numpy as np
import pandas as pd
import xgboost
from xgboost import plot_importance
from sklearn.ensemble import GradientBoostingClassifier as GBDT_C
from sklearn.linear_model import SGDClassifier as SGD_C
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from matplotlib import pyplot as plt

channel_dict = dict()
device_dict = dict()
os_dict = dict()
app_dict = dict()


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


def filted_train_data(df_train):
    df_pos = df_train.loc[df_train['is_attributed'] == 1]
    df_neg = df_train.loc[df_train['is_attributed'] != 1]
    pos_size = len(df_pos)
    print("size of pos attributed: %d" % pos_size)
    neg_size = len(df_neg)
    print("size of neg attributed: %d" % neg_size)
    neg_train = df_neg.sample(n=pos_size*3)
    train_data = pd.concat([df_pos, neg_train]).sample(frac=1).reset_index(drop=True)
    return train_data


def feature_construct(df):
    # Make some new features with click_time column
    df['datetime'] = pd.to_datetime(df['click_time'])
    # df['dow'] = df['datetime'].dt.dayofweek
    # df["doy"] = df["datetime"].dt.dayofyear
    df["hod"] = df["datetime"].dt.hour
    #df["dteom"]    = df["datetime"].dt.daysinmonth - df["datetime"].dt.day
    df.drop(['click_time', 'datetime'], axis=1, inplace=True)

    # channel counts on ip
    channel_by_ip = df.groupby(['ip'])['channel'].count().reset_index()
    channel_by_ip.columns = ['ip', 'channels_on_ip']
    df = pd.merge(df, channel_by_ip, on='ip', how='left', sort=False)
    # merge['clicks_by_ip'] = merge['clicks_by_ip'].astype('uint16')
    # df.drop('ip', axis=1, inplace=True)
    return df


def cluster_tiny_id(df):
    channel = df.channel.value_counts(normalize=False)
    for i in range(len(channel)):
        if channel.values[i] > 1:
            channel_dict[channel.index[i]] = 1
    device = df.device.value_counts(normalize=False)
    for i in range(len(device)):
        if device.values[i] > 1:
            device_dict[device.index[i]] = 1
    os = df.os.value_counts(normalize=False)
    for i in range(len(os)):
        if os.values[i] > 1:
            os_dict[os.index[i]] = 1
    app = df.app.value_counts(normalize=False)
    for i in range(len(app)):
        if app.values[i] > 1:
            app_dict[app.index[i]] = 1


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
    # print(cross_val_score(gbdt_model, train_data, label, cv=3, scoring='accuracy'))

    gbdt_model.fit(X=train_data, y=label)
    return gbdt_model


def xgb_train(train_file):
    df_train = pd.read_csv(train_file, nrows=1000000)
    # analysis click time
    # df_train['click_time'] = pd.to_datetime(df_train['click_time'])
    # df_train['click_time_D'] = df_train['click_time'].dt.to_period("D")
    # print("click_time limit to Day: ")
    # print(df_train['click_time_D'].unique())
    # df_train['click_time_H'] = df_train['click_time'].dt.to_period("H")
    # print("click_time limit to hour: ")
    # print(df_train['click_time_H'].unique())
    # end
    df_train = filted_train_data(df_train)
    df_train = feature_construct(df_train)
    print(len(channel_dict), len(device_dict))
    cluster_tiny_id(df_train)
    print(len(channel_dict), len(device_dict))

    label = np.array(df_train['is_attributed'], dtype=np.int32)
    df_train.drop(['attributed_time', 'is_attributed'], axis=1, inplace=True)
    print(df_train.head(10))

    train_data = np.array(df_train, dtype=np.int32)
    for i in range(train_data.shape[0]):
        if train_data[i, 4] not in channel_dict:
            train_data[i, 4] = -1
        if train_data[i, 2] not in device_dict:
            train_data[i, 2] = -1
        if train_data[i, 1] not in app_dict:
            train_data[i, 1] = -1
        if train_data[i, 3] not in os_dict:
            train_data[i, 3] = -1

    xgb_model = xgboost.XGBClassifier(max_depth=5,
                                      learning_rate=0.05,
                                      n_estimators=250,
                                      max_leaf_nodes=8,
                                      min_samples_split=6,
                                      njobs=4)
    print("xgboost cross score:")
    # print(cross_val_score(xgb_model, train_data, label, cv=3, scoring='accuracy'))

    xgb_model.fit(X=train_data, y=label)
    return xgb_model


def test(model, test_file):
    df_test_ori = pd.read_csv(test_file)

    df_test = feature_construct(df_test_ori)
    df_test = df_test.drop(['click_id'], axis=1, inplace=False)
    print(df_test.head(10))

    test_data = np.array(df_test, dtype=np.int32)
    for i in range(test_data.shape[0]):
        if test_data[i, 4] not in channel_dict:
            test_data[i, 4] = -1
        if test_data[i, 2] not in device_dict:
            test_data[i, 2] = -1
        if test_data[i, 1] not in app_dict:
            test_data[i, 1] = -1
        if test_data[i, 3] not in os_dict:
            test_data[i, 3] = -1

    y_ = model.predict_proba(test_data)
    print(y_[:3])
    size = (y_.shape)[0]
    result = pd.DataFrame({'click_id': df_test_ori['click_id'],
                           'is_attributed': y_[:,1].reshape(size).astype(np.float32)})
    print(result.head(10))
    result.to_csv("./data/result_test_xgb.csv", index=False)


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
    # df_test_ori = pd.read_csv(test_file)
    # analysis click time
    # df_test_ori['click_time'] = pd.to_datetime(df_test_ori['click_time'])
    # df_test_ori['click_time_D'] = df_test_ori['click_time'].dt.to_period("D")
    # print("click_time limit to Day: ")
    # print(df_test_ori['click_time_D'].unique())
    # df_test_ori['click_time_H'] = df_test_ori['click_time'].dt.to_period("H")
    # print("click_time limit to Hour: ")
    # print(df_test_ori['click_time_H'].unique())

    #model = train(train_file)
    model = xgb_train(train_file)
    joblib.dump(model, 'save/xgb_model.pkl')
    plot_importance(model)
    plt.gcf().savefig('feature_importance_xgb.png')
    plt.show()
    importance = model.booster().get_score(importance_type='weight')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    print("feature importance:")
    print(df)
    #model = batch_train(train_file)
    print("Begin predict:")
    test(model, test_file)
