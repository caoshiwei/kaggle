#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/2 17:42
# @Author  : Cao Shiwei
# @File    : test.py


import numpy as np
import pandas as pd
import paddle.v2 as paddle

import os

with_gpu = os.getenv('WITH_GPU', '0') != '0'


def convolutional_neural_network(img):
    # first conv layer
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # second conv layer
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # fully-connected layer
    predict = paddle.layer.fc(
        input=conv_pool_2, size=10, act=paddle.activation.Softmax())
    return predict


def creat_reader(filenm, is_pred):
    def reader():
        pd_data = pd.read_csv(filenm)
        ori_data = np.array(pd_data, dtype=np.float32)
        for i in xrange(len(ori_data)):
            if is_pred:
                yield ori_data[i, 0:] / 255.0 * 2.0 - 1.0
            else:
                yield ori_data[i, 1:] / 255.0 * 2.0 - 1.0, int(ori_data[i, 0])
    return reader


def main():
    paddle.init(use_gpu=with_gpu, trainer_count=1)

    # define network topology
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(10))

    # Here we can build the prediction network in different ways. Please
    # choose one by uncomment corresponding line.
    # predict = softmax_regression(images)
    # predict = multilayer_perceptron(images)
    out_layer = convolutional_neural_network(images)
    cost = paddle.layer.classification_cost(input=out_layer, label=label)
    #train(cost, out_layer)
    predict("params_pass_4.tar", "./data/paddle_res", out_layer)


def train(cost, out_layer):

    parameters = paddle.parameters.create(cost)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1 / 128.0,
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    lists = []

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                trainer.save_parameter_to_tar(f)

            result = trainer.test(reader=paddle.batch(
                paddle.dataset.mnist.test(), batch_size=128))
            print "Test with Pass %d, Cost %f, %s\n" % (
                event.pass_id, result.cost, result.metrics)
            lists.append((event.pass_id, result.cost,
                          result.metrics['classification_error_evaluator']))

    train_reader = creat_reader("./data/train.csv", False)
    feeding = {'pixel' : 0,
               'label': 1}

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(train_reader, buf_size=8192),
            batch_size=128),
        event_handler=event_handler,
        feeding=feeding,
        num_passes=2)

    # find the best pass
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
    print 'The classification accuracy is %.2f%%' % (100 - float(best[2]) * 100)

    pd_data = pd.read_csv("./data/test.csv")
    test_data = []
    test_data = np.array(pd_data, dtype=np.float32)
    test_data = test_data
    print(test_data[0:5])

    probs = paddle.infer(output_layer=out_layer, parameters=parameters, input=test_data, feeding=feeding)
    lab = np.argsort(-probs)  # probs and lab are the results of one batch data
    print("Label is:")
    print(lab)


def predict(model_file, out_file, out_layer):

    with open(model_file, 'r') as mf:
        params = paddle.parameters.Parameters.from_tar(mf)

        pd_data = pd.read_csv("./data/test.csv")
        test_data = []
        for line in np.array(pd_data, dtype=np.float32):
            test_data.append((np.array(line, dtype=np.float32) / 255.0 * 2.0 - 1.0,))
        # print(test_data[0:5])
        #input = test_data.tolist()
        feeding = {'pixel': 0}
        # test_data = []
        # test_reader = creat_reader("./data/test.csv", True)
        # for line in test_reader():
        #     test_data.append()
        # # print(test_data[0:2])
        #
        # print(test_data[0:3])
        probs = paddle.infer(
            output_layer=out_layer, parameters=params, input=test_data, feeding=feeding)
        lab = np.argsort(-probs)  # probs and lab are the results of one batch data
        print("Label is:")
        print(lab[:,0])
        pred_y = lab[:,0]
        result = pd.DataFrame({'ImageId': range(1, len(pred_y) + 1),
                               'Label': pred_y.astype(np.int32)})
        result.to_csv(out_file, index=False)
if __name__ == '__main__':
    main()

