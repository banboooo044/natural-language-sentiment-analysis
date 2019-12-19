import sys, os
sys.path.append('../')

import numpy as np
import pandas as pd
from src.runner import Runner
from src.model_GaussNB import ModelGaussNB

if __name__ == '__main__':
    params = {
        'priors' : None, 
        'var_smoothing' : 1e-09
    }
    params_NB = dict(params)

    # 特徴量を指定して実行
    feature = "bow"
    runner = Runner(run_name='GNB1', model_cls=ModelGaussNB, features=feature, params=params_NB)

    # 1回だけ実行
    # runner.train_fold(0)
    # クロスバリデーションで実行
    runner.run_train_cv()
