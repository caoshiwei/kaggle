#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/17 11:07
# @Author  : Cao Shiwei
# @File    : train.py.py

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from keras.layers import Embedding, Conv1D, Dense, SpatialDropout1D, Input
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.models import Model

from sklearn.metrics import roc_auc_score

import os
os.environ['OMP_NUM_THREADS'] = '4'


EMBEDDING_FILE = '../input/crawl-300d-2M.vec'

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

X_train = train_df[["question1", "question2"]].fillna("null").values
X_test = test_df[["question1", "question2"]].fillna("null").values

NUM_WORDS=100000
token_maker = Tokenizer(num_words=NUM_WORDS)
token_maker.fit_on_texts(np.append(X_train.reshape(-1), X_test.reshape(-1)))

# print(len(token_maker.word_counts))
train_tokens1 = token_maker.texts_to_sequences(X_train[:,0])
train_tokens2 = token_maker.texts_to_sequences(X_train[:,1])
test_tokens1 = token_maker.texts_to_sequences(X_test[:,0])
test_tokens2 = token_maker.texts_to_sequences(X_test[:,1])

max_len = 150
X_train1 = pad_sequences(train_tokens1, maxlen=max_len)
X_train2 = pad_sequences(train_tokens2, maxlen=max_len)
X_test1 = pad_sequences(test_tokens1, maxlen=max_len)
X_test2 = pad_sequences(test_tokens2, maxlen=max_len)

labels = train_df["is_duplicate"].values
print(labels.shape)
labels = labels.reshape(-1,1)
#labels = pad_sequences(labels.reshape(-1,1), maxlen=2)
print(labels.shape)

print("tokenizer over, begin get embed index:")


def get_ceofs(word, *arr):
    return word, np.asarray(arr, dtype=np.float32)


embedding_idx = dict(get_ceofs(*line.rstrip().rsplit(' ')) for line in open(EMBEDDING_FILE))

word_idx = token_maker.word_index
nb_words = min(NUM_WORDS, len(word_idx))

embed_size = 300
embedding_matrix = np.zeros((nb_words, embed_size))

for word, idx in word_idx.items():
    if idx >= NUM_WORDS:
        continue
    embed_vec = embedding_idx.get(word)
    if embed_vec is not None:
        embedding_matrix[idx] = embed_vec

print("embed index over, begin train:")


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch %d - score: %.6f \n" % (epoch+1, score))


def get_model():
    inp1 = Input(shape=(max_len,))
    inp2 = Input(shape=(max_len,))
    x1 = Embedding(NUM_WORDS, embed_size, weights=[embedding_matrix], trainable=False)(inp1)
    x2 = Embedding(NUM_WORDS, embed_size, weights=[embedding_matrix], trainable=False)(inp2)

    x1 = SpatialDropout1D(0.1)(x1)
    x2 = SpatialDropout1D(0.1)(x2)

    x1 = Conv1D(128, 3)(x1)
    x1_m_p = GlobalMaxPooling1D()(x1)
    x1_avg_p = GlobalAveragePooling1D()(x1)
    x1_conc = concatenate([x1_m_p, x1_avg_p])
    x1 = Dense(128, activation='relu')(x1_conc)

    x2 = Conv1D(128, 3)(x2)
    x2_m_p = GlobalMaxPooling1D()(x2)
    x2_avg_p = GlobalAveragePooling1D()(x2)
    x2_conc = concatenate([x2_m_p, x2_avg_p])
    x2 = Dense(128, activation='relu')(x2_conc)

    x = concatenate([x1, x2])
    outp = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[inp1, inp2], outputs=outp)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = get_model()
print(model.summary())
batch_size = 32
epochs = 2
model.fit(x=[X_train1, X_train2], y=labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save('./save/cur_model.h5')

submission = pd.read_csv('../input/sample_submission.csv')
y_pred = model.predict(x=[X_test1, X_test2], batch_size=1024)
submission[["is_duplicate"]] = y_pred
submission.to_csv('submission.csv', index=False)
