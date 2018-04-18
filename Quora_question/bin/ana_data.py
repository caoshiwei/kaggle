#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 19:35
# @Author  : Cao Shiwei
# @File    : ana_data.py

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

train_df.head()

test_df.head()

train_df.isnull().any(),test_df.isnull().any()
X_train = train_df[["question1", "question2"]].fillna("null").values
X_test = test_df[["question1", "question2"]].fillna("null").values

is_dup = train_df["is_duplicate"].value_counts()
print(is_dup)

num_words=100000
token_maker = Tokenizer(num_words=num_words)
token_maker.fit_on_texts(np.append(X_train.reshape(-1), X_test.reshape(-1)))

# print(len(token_maker.word_counts))
train_tokens1 = token_maker.texts_to_sequences(X_train[:,0])
train_tokens2 = token_maker.texts_to_sequences(X_train[:,1])
test_tokens1 = token_maker.texts_to_sequences(X_test[:,0])
test_tokens2 = token_maker.texts_to_sequences(X_test[:,1])

sentences_len = [len(sentence) for sentence in X_train]

plt.hist(sentences_len, bins=np.arange(0, 410, 10))
plt.show()

max_len = 150
X_train1 = pad_sequences(train_tokens1, maxlen=max_len)
X_train2 = pad_sequences(train_tokens2, maxlen=max_len)
X_test1 = pad_sequences(test_tokens1, maxlen=max_len)
X_test2 = pad_sequences(test_tokens2, maxlen=max_len)

from keras.layers import Input, Embedding, SpatialDropout1D, Conv1D, Dense
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.models import Model


labels = train_df["is_duplicate"].values
print(labels.shape)
labels = pad_sequences(labels)
print(labels.shape)

inp1 = Input(shape=(max_len,))
inp2 = Input(shape=(max_len, ))
embed_size = 256
x1 = Embedding(num_words, embed_size)(inp1)
x2 = Embedding(num_words, embed_size)(inp2)

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
outp = Dense(2, activation='sigmoid')(x)

model = Model(inputs=[inp1, inp2], outputs=outp)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 32
epochs = 2
model.fit(x=[X_train1, X_train2], y=labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True)


