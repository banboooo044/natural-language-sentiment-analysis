import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

from src.runner import Runner
from src.util import Logger

logger = Logger()

def makefig(result):
    sns.set_style("whitegrid")
    ax = sns.boxenplot(data = result, width=0.4)
    ax.set_ylabel('Accuracy', size=14)
    ax.tick_params(labelsize=14)
    plt.savefig(f'../model/tuning/{NAME}-logis.png',dpi=300)

if __name__ == '__main__':
    base_params = {
        'multi_class' :'multinomial', 
        'solver' : 'saga',
        'penalty' : 'l2', 
        'dual' :False, 
        'tol' :0.0001,
        'fit_intercept' : True, 
        'intercept_scaling' : 1, 
        'class_weight' : None, 
        'random_state' : 71, 
        'max_iter' : 500, 
        'verbose' : 1, 
        'warm_start' : False,
        'n_jobs' : None,
        'l1_ratio' : None
    }

    params_logistic = dict(base_params)
    param_grid_ = { 'C' : [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2 ]  }

    features = [ "bow", "tf-idf" ]
    NAME = ":".join(features)

    results = [ ]
    for name in features:
        x = Runner.load_x_train(name)
        y = Runner.load_y_train()
        model = ModelLogistic(name, **params_logistic)

        # GridSearchで最良なパラメータ探索
        search = GridSearchCV( model, cv=6, param_grid=param_grid_ , return_train_score=True )
        search.fit(x, y)

        # 探索結果を保存
        results.append( (search, name) )
        logger.info(f'{name} - bestscore : {search.best_score_} - result :{search.cv_results_["mean_test_score"]}')
    # 全ての探索結果をファイルに書き込み
    res = pd.DataFrame.from_dict(
        { name : search.cv_results_["mean_test_score"] for search, name in results }, 
        orient='index', 
        columns=param_grid_['C']
    )
    for search, name in results:
        logger.info(f'{name} - bestscore : {search.best_score_}')
    
    res.to_csv(f'../model/tuning/{NAME}-logis.csv')

    # 箱ひげ図もかく
    makefig(res)
