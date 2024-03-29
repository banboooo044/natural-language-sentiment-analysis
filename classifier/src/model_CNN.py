import os,sys
sys.path.append('../')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.callbacks import EarlyStopping
from keras.layers import SpatialDropout1D, Bidirectional, Input
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import ReLU, PReLU
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from gensim.models import KeyedVectors

from src.model import Model
from src.util import Util


import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class ModelCNN(Model):
    def __init__(self, run_fold_name, **params):
        super().__init__(run_fold_name, params)

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        """ 
            tr_x : List[str] (example.) [ "I am happy", "hello" ]
            tr_y : List[label]
            embedding_model : gensim.models.KeyedVectors Object
        """
        # scaling
        validation = va_x is not None

        # パラメータ
        nb_classes = 5
        embedding_dropout = self.params['embedding_dropout']
        lstm_dropout = self.params['lstm_dropout']
        lstm_recurrent_dropout = self.params['recurrent_dropout']
        hidden_layers = int(self.params['hidden_layers'])
        hidden_units = int(self.params['hidden_units'])
        hidden_activation = self.params['hidden_activation']
        hidden_dropout = self.params['hidden_dropout']
        batch_norm = self.params['batch_norm']
        optimizer_type = self.params['optimizer']['type']
        optimizer_lr = self.params['optimizer']['lr']
        batch_size = int(self.params['batch_size'])
        nb_epoch = int(self.params['nb_epoch'])
        embedding_model = self.params['embedding_model']
        bidirectional = self.params['Bidirectional']
        use_pre_embedding = not (embedding_model is None)

        # using keras tokenizer here
        token = Tokenizer(num_words=None)
        max_len = 70
        if validation:
            token.fit_on_texts(list(tr_x) + list(va_x))
        else:
            token.fit_on_texts(list(tr_x))

        xtrain_seq = token.texts_to_sequences(tr_x)
        tr_x = pad_sequences(xtrain_seq, maxlen=max_len)
        tr_y = np_utils.to_categorical(tr_y, num_classes=5)

        if validation:
            xvalid_seq = token.texts_to_sequences(va_x)
            va_x = pad_sequences(xvalid_seq, maxlen=max_len)
            va_y = np_utils.to_categorical(va_y, num_classes=5)

        word_index = token.word_index

        if use_pre_embedding:
            # create an embedding matrix
            vector_dim = embedding_model.vector_size
            embedding_matrix = np.zeros((len(word_index) + 1, vector_dim))
            for word, i in tqdm(word_index.items()):
                embedding_vector = embedding_model.wv[word]
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            
        inputs = Input(shape=(max_len, ))
        # input layer
        if use_pre_embedding:
            self.model.add(Embedding(
                    input_dim=len(word_index) + 1, 
                    output_dim=vector_dim,
                    input_length=max_len,
                    weights=[embedding_matrix],
                    trainable=False))
        else:
            self.model.add(Embedding(input_dim=len(word_index) + 1, 
                    output_dim=300,
                    input_length=max_len))

        self.model.add(SpatialDropout1D(embedding_dropout))
        if bidirectional:
            self.model.add(Bidirectional(LSTM(300, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)))
        else:
            self.model.add(LSTM(100, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout))
        # 中間層
        for i in range(hidden_layers):
            self.model.add(Dense(hidden_units))
            if batch_norm == 'before_act':
                self.model.add(BatchNormalization())
            if hidden_activation == 'prelu':
                self.model.add(PReLU())
            elif hidden_activation == 'relu':
                self.model.add(ReLU())
            else:
                raise NotImplementedError
            self.model.add(Dropout(hidden_dropout))

        # 出力層
        self.model.add(Dense(nb_classes, activation='sigmoid'))

        # オプティマイザ
        if optimizer_type == 'sgd':
            optimizer = SGD(lr=optimizer_lr, decay=1e-6, momentum=0.9, nesterov=True)
        elif optimizer_type == 'adam':
            optimizer = Adam(lr=optimizer_lr, beta_1=0.9, beta_2=0.999, decay=0.)
        else:
            raise NotImplementedError

        # 目的関数、評価指標などの設定
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # エポック数、アーリーストッピング
        # あまりepochを大きくすると、小さい学習率のときに終わらないことがあるので注意
        patience = 20
        # 学習の実行
        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                            verbose=1, restore_best_weights=True)
            history = self.model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=batch_size, verbose=2,
                                validation_data=(va_x, va_y), callbacks=[early_stopping])
        else:
            history = self.model.fit(tr_x, tr_y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2)

    def predict(self, te_x):
        te_x = self.scaler.transform(te_x)
        y_pred = self.model.predict(te_x)
        return y_pred

    def score(self, te_x, te_y):
        y_pred = self.predict(te_x)
        return accuracy_score(te_y, np.argmax(y_pred, axis=1))

    def save_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}-scaler.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        Util.dump(self.scaler, scaler_path)

    def load_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}-scaler.pkl')
        self.model = load_model(model_path)
        self.scaler = Util.load(scaler_path)

