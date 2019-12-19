# 基本となるパラメータ

import sys, os
sys.path.append('../')

import numpy as np
import pandas as pd
from src.runner import Runner
from src.model_LSTM import ModelLSTM

from gensim.models import KeyedVectors


if __name__ == '__main__':
    # LSTM
    params = {
        'embedding_dropout' : 0.3,
        'lstm_dropout' : 0.3,
        'recurrent_dropout' : 0.3,
        'hidden_layers': 3,
        'hidden_units': 256,
        'hidden_activation': 'prelu',
        'hidden_dropout': 0.5,
        'batch_norm': 'before_act',
        'optimizer': {'type': 'adam', 'lr': 0.005},
        'batch_size': 100,
        'nb_epoch' : 500,
        'embedding_model' : None,
        'Bidirectional' : False,
    }
    #  双方向LSATM
    #params = {
    

    #}

    # fasttext.bin は compress.py　でボキャブラリを圧縮したファイル
    params['embedding_model'] = KeyedVectors.load_word2vec_format('./fasttext.bin', binary=True)
    params_LSTM = dict(params)
    
    # features には必ず raw_textを指定
    runner = Runner(run_name='LSTM1', model_cls=ModelLSTM, features="raw_text", params=params_LSTM)

    # 1回だけ実行
    # runner.train_fold(0)
    # クロスバリデーションで実行
    runner.run_train_cv()
