import sys
sys.path.append('../')
import numpy as np
import pandas as pd

from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, Trials

# 学習データを学習データとバリデーションデータに分ける
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt
import seaborn as sns

from src.util import Logger
from src.model_MLP import ModelMLP
from src.runner import Runner

import gc
gc.collect()
logger = Logger()

# 目的関数
def objective(params):
    global base_params
    # base parameter のパラメータを探索パラメータに更新する
    base_params.update(params)

    # モデルオブジェクト生成 & Train
    model = ModelMLP("MLP", **base_params)
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
    # 基本となるパラメータ
    base_params = {
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

    features = [
        "tf-idf","n-gram","n-gram-tf-idf"
    ]

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
    res.to_csv(f'../model/tuning/MLP-{NAME}.csv')
