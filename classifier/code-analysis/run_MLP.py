import sys, os
sys.path.append('../')

import numpy as np
import pandas as pd
from src.runner import Runner
from src.model_MLP import ModelMLP

if __name__ == '__main__':
    params = {
        'input_dropout': 0.0,
        'hidden_layers': 3,
        'hidden_units': 96,
        'hidden_activation': 'relu',
        'hidden_dropout': 0.2,
        'batch_norm': 'before_act',
        'optimizer': {'type': 'adam', 'lr': 0.001},
        'batch_size': 64,
        'nb_epoch' : 1000
    }
    #### Best Parameters
    bow =              {'batch_norm': 'before_act', 'batch_size': 96.0, 'hidden_activation': 'relu', 'hidden_dropout': 0.3, 'hidden_layers': 2.0, 'hidden_units': 160.0, 'input_dropout': 0.05, 'optimizer': {'lr': 0.0015, 'type': 'adam'}}
    #tf-idf =          {'batch_norm': 'no', 'batch_size': 192.0, 'hidden_activation': 'prelu', 'hidden_dropout': 0.0, 'hidden_layers': 2.0, 'hidden_units': 192.0, 'input_dropout': 0.05, 'optimizer': {'lr': 0.0018, 'type': 'adam'}}
    #word2vec_mean =   {'batch_norm': 'before_act', 'batch_size': 256.0, 'hidden_activation': 'prelu', 'hidden_dropout': 0.25, 'hidden_layers': 2.0, 'hidden_units': 128.0, 'input_dropout': 0.15, 'optimizer': {'lr': 0.00037, 'type': 'adam'}}
    #word2vec_max =    {'batch_norm': 'no', 'batch_size': 32.0, 'hidden_activation': 'relu', 'hidden_dropout': 0.3, 'hidden_layers': 3.0, 'hidden_units': 160.0, 'input_dropout': 0.15, 'optimizer': {'lr': 0.00016, 'type': 'adam'}}
    #word2vec_concat = {'batch_norm': 'before_act', 'batch_size': 32.0, 'hidden_activation': 'prelu', 'hidden_dropout': 0.25, 'hidden_layers': 2.0, 'hidden_units': 96.0, 'input_dropout': 0.15, 'optimizer': {'lr': 0.00075, 'type': 'sgd'}}
    #word2vec_hier =   {'batch_norm': 'no', 'batch_size': 96.0, 'hidden_activation': 'prelu', 'hidden_dropout': 0.25, 'hidden_layers': 3.0, 'hidden_units': 256.0, 'input_dropout': 0.2, 'optimizer': {'lr': 0.0024, 'type': 'sgd'}}
    #fasttext_mean =   {'batch_norm': 'before_act', 'batch_size': 224.0, 'hidden_activation': 'relu', 'hidden_dropout': 0.3, 'hidden_layers': 2.0, 'hidden_units': 192.0, 'input_dropout': 0.2, 'optimizer': {'lr': 0.0032, 'type': 'sgd'}}
    #fasttex_max =     {'batch_norm': 'no', 'batch_size': 160.0, 'hidden_activation': 'prelu', 'hidden_dropout': 0.25, 'hidden_layers': 3.0, 'hidden_units': 128.0, 'input_dropout': 0.2, 'optimizer': {'lr': 0.00016, 'type': 'adam'}}
    #fasttext_concat = {'batch_norm': 'before_act', 'batch_size': 192.0, 'hidden_activation': 'prelu', 'hidden_dropout': 0.6, 'hidden_layers': 2.0, 'hidden_units': 224.0, 'input_dropout': 0.15, 'optimizer': {'lr': 0.00048, 'type': 'adam'}}
    #fasttext_hier =   {'batch_norm': 'no', 'batch_size': 64.0, 'hidden_activation': 'prelu', 'hidden_dropout': 0.3, 'hidden_layers': 2.0, 'hidden_units': 128.0, 'input_dropout': 0.2, 'optimizer': {'lr': 0.00025, 'type': 'adam'}}
    #doc2vec-dbow =    {'batch_norm': 'no', 'batch_size': 96.0, 'hidden_activation': 'prelu', 'hidden_dropout': 0.25, 'hidden_layers': 4.0, 'hidden_units': 160.0, 'input_dropout': 0.2, 'optimizer': {'lr': 0.0017, 'type': 'sgd'}}
    #doc2vec-dmpv =    {'batch_norm': 'before_act', 'batch_size': 192.0, 'hidden_activation': 'relu', 'hidden_dropout': 0.25, 'hidden_layers': 4.0, 'hidden_units': 224.0, 'input_dropout': 0.2, 'optimizer': {'lr': 0.0040, 'type': 'sgd'}},
    #doc2vec-concat =  {'batch_norm': 'no', 'batch_size': 160.0, 'hidden_activation': 'relu', 'hidden_dropout': 0.25, 'hidden_layers': 3.0, 'hidden_units': 256.0, 'input_dropout': 0.05, 'optimizer': {'lr': 0.0025, 'type': 'sgd'}}
    #sdv =             {'batch_norm': 'before_act', 'batch_size': 192.0, 'hidden_activation': 'relu', 'hidden_dropout': 0.25, 'hidden_layers': 3.0, 'hidden_units': 256.0, 'input_dropout': 0.2, 'optimizer': {'lr': 0.0029, 'type': 'sgd'}}

    params.update(bow)
    params_MLP = dict(params)

    # MLPで予測
    feature = "bow"
    runner = Runner(run_name='MLP1', model_cls=ModelMLP, features=feature, params=params_MLP)
    runner.run_train_cv()
