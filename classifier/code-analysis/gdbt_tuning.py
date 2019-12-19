import sys
sys.path.append('../')
import numpy as np
import pandas as pd

from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, Trials

import matplotlib.pyplot as plt
import seaborn as sns

from src.util import load_x_train, load_y_train, Logger
from src.model_lgb import ModelLGB


logger = Logger()

import gc
gc.collect()
# 学習データを学習データとバリデーションデータに分ける
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

def objective(params):
    global param
    params_tmp = {
        'num_leaves': int(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }
    params.update(params_tmp)
    # パラメータセットを指定したときに最小化すべき関数を指定する
    # モデルのパラメータ探索においては、モデルにパラメータを指定して学習・予測させた場合のスコアとする
    param.update(params)
    model = ModelLGB("LGB", **param)
    model.train(tr_x, tr_y, va_x, va_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    print(f'params: {params}, logloss: {score:.4f}')
    # 情報を記録しておく
    history.append((params, score))
    del model

    return {'loss': score, 'status': STATUS_OK}

if __name__ == '__main__':
    # ベースラインのパラメータ
    param = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class' : 5,
        'silent': 1,
        'random_state': 71,
        'num_boost_round': 1000,
        'early_stopping_rounds': 10,
        'n_estimator' : 500
    }

    # パラメータの探索範囲
    param_space = {
        'learning_rate':    hp.choice('learning_rate', [0.001, 0.01, 0.05, 0.08]),
        'num_leaves':       hp.quniform('num_leaves', 8, 64, 2),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 0.8),
        'subsample':        hp.uniform('subsample', 0.5, 1),
    }

    #features = [
    #    "bow","bow_nva","bow_tf-idf","term_2-gram","term_3-gram","word2vec_mean","word2vec_pre_mean",
    #    "word2vec_fine-tuning", "doc2vec", "scdv", "bert"
    #]
    #features = [
    #    "bow", "tf-idf","n-gram","n-gram-tf-idf"
    #]
    #features = [
    #    "word2vec_mean", "word2vec_max", "word2vec_concat", "word2vec_hier", "fasttext_mean", "fasttext_max", "fasttext_concat", "fasttext_hier"
    #]
    features = [
        'sdv'
    ]
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
    res.to_csv(f'../model/tuning/LGB-{NAME}.csv')