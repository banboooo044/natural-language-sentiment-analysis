# 基本となるパラメータ

import sys, os
sys.path.append('../')

import numpy as np
import pandas as pd
from src.runner import Runner
from src.model_LSTM import ModelLSTM

from gensim.models import KeyedVectors

# "bow", "bow_nva", "bow_tf-idf", "term_2-gram", "term_3-gram", "word2vec_mean", "word2vec_pre_mean", "word2vec_fine-tuning-iter25", "word2vec_fine-tuning-iter5", "doc2vec"

if __name__ == '__main__':
    params = {
        'embedding_dropout' : 0.3,
        'lstm_dropout' : 0.3,
        'recurrent_dropout' : 0.3,
        'hidden_layers': 3,
        'hidden_units': 1024,
        'hidden_activation': 'relu',
        'hidden_dropout': 0.8,
        'batch_norm': 'before_act',
        'optimizer': {'type': 'adam', 'lr': 0.001},
        'batch_size': 512,
        'nb_epoch' : 100,
        'embedding_model' : None,
        'Bidirectional' : False
    }

    # NORMALLSTM
    """#23
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
    
    """

    params['embedding_model'] = KeyedVectors.load("../vec/word2vec_fine-tuning-iter5.model")
    params_LSTM = dict(params)
    
    runner = Runner(run_name='LSTM1', model_cls=ModelLSTM, features="raw_text", params=params_LSTM)
    runner.run_train_cv()
    #runner.run_train_cv([ 100, 500, 1000, 2000, 5000, 6000, 7000, 8000])
    #runner.run_predict_cv()
