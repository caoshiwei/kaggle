import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Dropout, Conv1D
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, TimeDistributed
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, History

import warnings
warnings.filterwarnings('ignore')

from keras import backend
import tensorflow as tf
thread_jobs = 4
config = tf.ConfigProto(intra_op_parallelism_threads=thread_jobs, \
                        inter_op_parallelism_threads=thread_jobs, \
                        allow_soft_placement=True, \
                        device_count = {'CPU': thread_jobs})
session = tf.Session(config=config)
K.set_session(session)


EMBEDDING_FILE = '../input/crawl-300d-2M.vec'

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')

X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values


max_features = 30000
maxlen = 200
embed_size = 300
print("read data done, begin tokenizer:")

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print("tokenizer over, begin get embed index:")

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

print("embed index over, begin train:")


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    text_inp = SpatialDropout1D(0.2)(x)

    text_conv_1 = Conv1D(128, 4)(text_inp)
    GRU_1 = Bidirectional(GRU(64, return_sequences=True))(text_conv_1)
    #x = TimeDistributed(Dense(128))(x)
    rec_1 = concatenate([text_conv_1, GRU_1])

    text_conv_2 = Conv1D(128, 4)(rec_1)
    GRU_2 = Bidirectional(GRU(64, return_sequences=True))(text_conv_2)
    rec_out = concatenate([text_conv_2, GRU_2])
    # x = TimeDistributed(Dense(64))
    avg_pool = GlobalAveragePooling1D()(rec_out)
    max_pool = GlobalMaxPooling1D()(rec_out)
    conc = concatenate([avg_pool, max_pool])

    conc = Dense(128, activation='relu')(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = get_model()
print(model.summary())


def data_genrator(X, y, batch_size=32):
    data_size = len(y)
    loop_size = data_size // batch_size
    while 1:
        for i in range(loop_size):
            yield X[i: i + batch_size, : ], y[i : i + batch_size, :]


batch_size = 64
epochs = 100

early_stopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
chk_poi_file = "./save/weights.{epoch:02d-{val_loss:.2f}}.hdf5"
check_point = ModelCheckpoint(chk_poi_file, monitor='val_loss', verbose=1, period=20)

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

# hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
#                  callbacks=[RocAuc, check_point, early_stopper], verbose=1, steps_per_epoch=5000)
hist = model.fit_generator(data_genrator(X_tra, y_tra), steps_per_epoch=2000, epochs=100,
                           validation_data=(X_val, y_val), callbacks=[RocAuc, check_point, early_stopper], verbose=1)

# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)

y_pred = model.predict(x_test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)