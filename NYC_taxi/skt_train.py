#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/8 14:58
# @Author  : Cao Shiwei
# @File    : skt_train.py

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score

import datetime
import random


PASSER_MAX = 9


def build_time_dict(date_times):
    date_dict = dict()
    date_s = set()
    date_dict['unk'] = 0
    # time_s = set()
    for date_time in date_times:
        items = date_time.strip().split(' ')
        if len(items) == 2:
            date_s.add(items[0])
            #time_s.add(items[1])

    for i,_ in enumerate(date_s):
        date_dict[i] = _ + 1
    return date_dict, len(date_dict)


def calc_distance(latitude1, longitude1, latitude2, longitude2):
    dis_lat = np.abs(latitude1 - latitude2)
    dis_long = np.abs(longitude1 - longitude2)
    R = 6371
    a1 = np.sin(dis_lat/2.0)**2
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt((1-a1)))
    latitude_dis = R * c1
    a2 = np.sin(dis_long / 2.0)**2
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt((1-a2)))
    longitude_dis = R * c2
    return np.abs(latitude_dis) + np.abs(longitude_dis)


def feature_eng(data, train):
    #date_dict, d_size = build_time_dict(data['pickup_datetime'])
    week_list = list()
    hour_list = list()
    min_list = list()
    for date_item in data['pickup_datetime']:
        items = date_item.strip().split(' ')
        if len(items) == 2:
            y, m, d = items[0].split('-')
            week = datetime.datetime(int(y), int(m), int(d)).weekday()
            week_list.append(week)
            h, min, sec = items[1].split(':')
            hour_list.append(int(h))
            min_list.append(int(min))
    print(week_list[0:5])
    data['date_week'] = np.array(week_list) / 7.0
    print(data['date_week'][0:5])
    data['date_hour'] = np.array(hour_list) / 24.0
    data['date_min'] = np.array(min_list) / 59.0

    distance = list()
    for i in range(len(np.array(data['pickup_longitude']))):
        manhattan_dis = calc_distance(data['pickup_latitude'][i],
                                      data['pickup_longitude'][i],
                                      data['dropoff_latitude'][i],
                                      data['dropoff_longitude'][i])
        distance.append(manhattan_dis)
    data['distance'] = distance

    data['vendor_id'] = data['vendor_id'] / 2.0
    data['passenger_count'] = data['passenger_count'] / PASSER_MAX
    data['pickup_longitude'] = data['pickup_longitude'] / 180.0
    data['pickup_latitude'] = data['pickup_latitude'] / 90.0
    data['dropoff_longitude'] = data['dropoff_longitude'] / 180.0
    data['dropoff_latitude'] = data['dropoff_latitude'] / 90.0
    data.loc[data['store_and_fwd_flag'] == 'N', 'store_and_fwd_flag'] = 0
    data.loc[data['store_and_fwd_flag'] == 'Y', 'store_and_fwd_flag'] = 1

    print(data.describe())
    if train:
        data_res = data.drop(['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration'], axis=1, inplace=False)
        label = data['trip_duration']
        return np.array(data_res, dtype=np.float32), np.array(label, dtype=np.int32)
    else:
        data_res = data.drop(['id', 'pickup_datetime'], axis=1, inplace=False)
        return np.array(data_res, dtype=np.float32)


def read_data(filenm, train):
    ori_data = pd.read_csv(filenm)
    return feature_eng(ori_data, train)


def train():
    X, label = read_data("./data/train.csv", True)
    datasize = X.shape[0]
    print(datasize)
    ids = random.sample(range(datasize), 100000)
    batch_x = X[ids]
    batch_labels = label[ids]
    print(X[0:5])
    print(label[0:5])
    svm_r = svm.SVR(kernel='rbf', C=1, gamma=0.01)

    #param_test = {"C": range(1,20,1)}
    #gsearch = GridSearchCV(estimator=svm_c, param_grid=param_test, scoring='accuracy', cv=5)
    #gsearch.fit(train_x, train_y)
    #print(gsearch.best_params_, gsearch.best_score_)
    svm_r.fit(X, label)
    print("svm score")
    #print(cross_val_score(svm_r, batch_x, batch_labels, cv=3, scoring='neg_mean_squared_log_error', n_jobs=-1))
    #print(cross_val_score(bagging_svm, train_x, train_y, cv=5, scoring='accuracy'))
    return svm_r


def predict(model, filenm):
    ori_data = pd.read_csv(filenm)
    X = feature_eng(ori_data, False)
    y_ = model.predict(X)
    result = pd.DataFrame({'id': ori_data['id'].as_matrix(),
                           'trip_duration': y_.astype(np.int32)})
    # print(result)
    result.to_csv("./data/result_test.csv", index=False)


def main():
    model = train()
    print("train over!")
    predict(model, "./data/test.csv")
    print("all over!")


if __name__ == "__main__":
    main()