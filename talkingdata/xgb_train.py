#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 11:46
# @Author  : Cao Shiwei
# @File    : talking_model.py

import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


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


def xgb_train(train_file, is_valid):
    df_train = pd.read_csv(train_file)

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
    params = {'eta': 0.3,
              'booster': 'gbtree',
              # 'gamma': 0.01, #default 0
              'tree_method': "hist",  #default 'auto'
              'grow_policy': "lossguide",
              'max_leaves': 1400,
              'max_depth': 0,
              'subsample': 0.9,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'min_child_weight': 0,
              'alpha': 4,
              'objective': 'binary:logistic',
              'scale_pos_weight': 9,
              'eval_metric': 'auc',
              'nthread': 8,
              'random_state': 99,
              'silent': True}

    if is_valid:
        # Get 10% of train dataset to use as validation
        x1, x2, y1, y2 = train_test_split(train_data, label, test_size=0.1, random_state=99)
        dtrain = xgboost.DMatrix(x1, y1)
        dvalid = xgboost.DMatrix(x2, y2)
        del x1, y1, x2, y2
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        xgboost.Booster.set_param()
        model = xgboost.train(params, dtrain, 200, watchlist, maximize=True, early_stopping_rounds=25, verbose_eval=5)
        del dvalid
    else:
        dtrain = xgboost.DMatrix(train_data, label)
        del train_data, label
        #gc.collect()
        watchlist = [(dtrain, 'train')]
        model = xgboost.train(params, dtrain, 30, watchlist, maximize=True, verbose_eval=1)
    print("xgboost cross score:")
    return model
    # print(cross_val_score(xgb_model, train_data, label, cv=3, scoring='accuracy'))


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

    dtest = xgboost.DMatrix(test_data)
    del test_data

    y_ = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    print(y_[:3])
    size = (y_.shape)[0]
    result = pd.DataFrame({'click_id': df_test_ori['click_id'],
                           'is_attributed': y_[:,1].reshape(size).astype(np.float32)})
    print(result.head(10))
    result.to_csv("./data/result_test_xgb.csv", index=False)


if __name__ == '__main__':
    train_file = "./data/train.csv"
    test_file = "./data/test.csv"

    model = xgb_train(train_file, True)
    #model = batch_train(train_file)
    print("Begin predict:")
    test(model, test_file)
