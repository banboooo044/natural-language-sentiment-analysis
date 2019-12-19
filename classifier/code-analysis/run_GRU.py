# 基本となるパラメータ

import sys, os
sys.path.append('../')

import numpy as np
import pandas as pd
from src.runner import Runner
from src.model_GRU import ModelGRU

from gensim.models import KeyedVectors

if __name__ == '__main__':
    params = {
        'embedding_dropout' : 0.3,
        'gru_dropout' : 0.3,
        'recurrent_dropout' : 0.3,
        'hidden_layers': 3,
        'hidden_units': 128,
        'hidden_activation': 'relu',
        'hidden_dropout': 0.3,
        'batch_norm': 'before_act',
        'optimizer': {'type': 'adam', 'lr': 0.001},
        'batch_size': 100,
        'nb_epoch' : 500,
        'embedding_model' : None
    }

    # fasttext.bin は compress.py　でボキャブラリを圧縮したファイル
    params['embedding_model'] = KeyedVectors.load_word2vec_format('./fasttext.bin', binary=True)
    params_GRU = dict(params)
    
    # features には必ず raw_textを指定
    runner = Runner(run_name='GRU1', model_cls=ModelGRU, features="raw_text", params=params_GRU)

    # 1回だけ実行
    # runner.train_fold(0)
    # クロスバリデーションで実行
    runner.run_train_cv()