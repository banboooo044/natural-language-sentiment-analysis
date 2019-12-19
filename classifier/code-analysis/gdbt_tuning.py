# Parameter tuning for LightGBM 
import sys
sys.path.append('../')
import numpy as np
import pandas as pd

from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, Trials

import matplotlib.pyplot as plt
import seaborn as sns

import gc
gc.collect()

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from src.runner import Runner
from src.util import Logger
from src.model_lgb import ModelLGB

logger = Logger()

# 目的関数
def objective(params):
    # params に次の探索パラメータが入る
    global base_params
    # 少数を丸める
    params_tmp = {
        'num_leaves': int(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }
    # base parameter のパラメータを探索パラメータに更新する
    params.update(params_tmp)
    base_params.update(params)

    # モデルオブジェクト生成 & Train
    model = ModelLGB("LGB", **base_params)
    model.train(tr_x, tr_y, va_x, va_y)
    # 予測
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    print(f'params: {params}, logloss: {score:.4f}')
    # 情報を記録しておく
    history.append((params, score))
    del model
    return {'loss': score, 'status': STATUS_OK}


if __name__ == '__main__':
    # ベースラインのパラメータ
    base_params = {
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

    features = [ "bow", "tf-idf" ]
    NAME = "-".join(features)
    result = { }

    
    # 1つの特徴量について探索するパラメータの組み合わせ数
    max_evals = 100
    # リストfeaturesの各特徴量に対して, 最良なパラメータを探索する
    for i, name in enumerate(features):
        train_x = Runner.load_x_train(name)
        train_y = Runner.load_y_train()
        skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=71)
        tr_idx, va_idx = list(skf.split(train_x, train_y))[0]
        tr_x, va_x = train_x[tr_idx], train_x[va_idx]
        tr_y, va_y = train_y[tr_idx], train_y[va_idx]

        # hyperoptによるパラメータ探索の実行
        trials = Trials()
        history = []
        fmin(objective, param_space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

        # 探索結果をログに出力
        history = sorted(history, key=lambda tpl: tpl[1])
        best = history[0]
        logger.info(f'{name} - best params:{best[0]}, score:{best[1]:.4f}')
        
        # 探索結果を記録
        for key, value in best[0].items():
            if key in result:
                result[key].append(value)
            else:
                result[key] = [value]

    # 全ての探索結果をファイルに書き込み
    res = pd.DataFrame.from_dict(
        result,
        orient='index',
        columns=features
    )
    res.to_csv(f'../model/tuning/LGB-{NAME}.csv')
