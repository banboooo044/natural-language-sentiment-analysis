import sys
sys.path.append('../')
import numpy as np
import pandas as pd

from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, Trials

import matplotlib.pyplot as plt
import seaborn as sns

from src.util import load_x_train, load_y_train, Logger
from src.model_MLP import ModelMLP

import gc
gc.collect()
logger = Logger()

# 学習データを学習データとバリデーションデータに分ける
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

def objective(params):
    global param
    param.update(params)
    model = ModelMLP("MLP", **param)
    model.train(tr_x, tr_y, va_x, va_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    print(f'params: {params}, logloss: {score:.4f}')
    # 情報を記録しておく
    history.append((params, score))
    del model

    return {'loss': score, 'status': STATUS_OK}

if __name__ == '__main__':
    # 基本となるパラメータ
    param = {
        'input_dropout': 0.0,
        'hidden_layers': 3,
        'hidden_units': 96,
        'hidden_activation': 'relu',
        'hidden_dropout': 0.2,
        'batch_norm': 'before_act',
        'optimizer': {'type': 'adam', 'lr': 0.001},
        'batch_size': 64,
        'nb_epoch': 100,
    }
    # 探索するパラメータの空間を指定する
    param_space = {
        'input_dropout': hp.quniform('input_dropout', 0, 0.2, 0.05),
        'hidden_layers': hp.quniform('hidden_layers', 2, 4, 1),
        'hidden_units': hp.quniform('hidden_units', 32, 256, 32),
        'hidden_activation': hp.choice('hidden_activation', ['prelu', 'relu']),
        'hidden_dropout': hp.quniform('hidden_dropout', 0, 0.3, 0.05),
        'batch_norm': hp.choice('batch_norm', ['before_act', 'no']),
        'optimizer': hp.choice('optimizer',
                           [{'type': 'adam',
                             'lr': hp.loguniform('adam_lr', np.log(0.00001), np.log(0.01))},
                            {'type': 'sgd',
                             'lr': hp.loguniform('sgd_lr', np.log(0.00001), np.log(0.01))}]),
        'batch_size': hp.quniform('batch_size', 32, 128, 32)
    }

    #features = [
    #    "bow","bow_nva","bow_tf-idf","term_2-gram","term_3-gram","word2vec_mean","word2vec_pre_mean",
    #    "word2vec_fine-tuning", "doc2vec", "scdv", "bert"
    #]
    features = [
        "tf-idf","n-gram","n-gram-tf-idf"
    ]
    #features = [
    #    "word2vec_mean", "word2vec_max", "word2vec_concat", "word2vec_hier", "fasttext_mean", "fasttext_max", "fasttext_concat", "fasttext_hier"
    #]

    NAME = "-".join(features)

    result = { }
    for i, name in enumerate(features):
        train_x = load_x_train(name)
        train_y = load_y_train(name)
        skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=71)
        tr_idx, va_idx = list(skf.split(train_x, train_y))[0]
        tr_x, va_x = train_x[tr_idx], train_x[va_idx]
        tr_y, va_y = train_y[tr_idx], train_y[va_idx]

        # hyperoptによるパラメータ探索の実行
        max_evals = 100
        trials = Trials()
        history = []
        fmin(objective, param_space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
        history = sorted(history, key=lambda tpl: tpl[1])
        best = history[0]
        logger.info(f'{name} - best params:{best[0]}, score:{best[1]:.4f}')
        
        for key, value in best[0].items():
            if key in result:
                result[key].append(value)
            else:
                result[key] = [value]
    res = pd.DataFrame.from_dict(
        result,
        orient='index',
        columns=features
    )
    res.to_csv(f'../model/tuning/MLP-{NAME}.csv')
