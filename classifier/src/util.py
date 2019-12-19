# 雑多プログラム
import datetime
import logging
import os

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from scipy.sparse import load_npz

class Util:
    @classmethod
    def dump(cls, value, path):
        """ モデルの保存 """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        """ モデルのロード """
        return joblib.load(path)

class Logger:
    """ ログをコンソール, ファイル保存する """
    def __init__(self):
        self.general_logger = logging.getLogger('general')
        self.result_logger = logging.getLogger('result')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler('../model/general.log')
        file_result_handler = logging.FileHandler('../model/result.log')
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        # 時刻をつけてコンソールとログに出力
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, scores):
        # 計算結果をコンソールと計算結果用ログに出力
        dic = dict()
        dic['name'] = run_name
        dic['score'] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f'score{i}'] = score
        self.result_ltsv(dic)

    def now_string(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_ltsv(self, dic):
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])

"""
def load_x_train(features, sparse=False):
    if features == "bow":
        matrix =load_npz('../vec/bow_train_x.npz').astype('float64')
    elif features == "n-gram":
        matrix = load_npz('../vec/n-gram_x.npz').astype('float64')
    elif features == "tf-idf":
        matrix = load_npz('../vec/tf-idf_x.npz').astype('float64')
    elif features == "n-gram-tf-idf":
        matrix = load_npz('../vec/n-gram-tf-idf_x.npz').astype('float64')
    elif features == "word2vec_mean":
        matrix = np.load('../vec/word2vec_x_mean.npy', allow_pickle = True)
    elif features == "word2vec_max":
        matrix = np.load('../vec/word2vec_x_max.npy', allow_pickle = True)
    elif features == "word2vec_concat":
        l = np.load('../vec/word2vec_x_mean.npy', allow_pickle = True)
        r = np.load('../vec/word2vec_x_max.npy', allow_pickle = True)
        matrix = np.hstack((l, r))
    elif features == "word2vec_hier":
        matrix = np.load('../vec/word2vec_x_hier.npy', allow_pickle = True)
    elif features == "fasttext_mean":
        matrix = np.load('../vec/fasttext_x_mean.npy', allow_pickle = True)
    elif features == "fasttext_max":
        matrix = np.load('../vec/fasttext_x_max.npy', allow_pickle = True)
    elif features == "fasttext_concat":
        l = np.load('../vec/fasttext_x_mean.npy', allow_pickle = True)
        r = np.load('../vec/fasttext_x_max.npy', allow_pickle = True)
        matrix = np.hstack((l, r))
    elif features == "fasttext_hier":
        matrix = np.load('../vec/fasttext_x_hier.npy', allow_pickle = True)
    elif features == "doc2vec-dbow":
        matrix = np.load('../vec/doc2vec_x.npy', allow_pickle=True)
    elif features == "doc2vec-dmpv":
        matrix = np.load('../vec/doc2vec-dmpv_x.npy', allow_pickle=True)
    elif features == "doc2vec-concat":
        l = np.load('../vec/doc2vec_x.npy', allow_pickle=True)
        r = np.load('../vec/doc2vec-dmpv_x.npy', allow_pickle=True)
        matrix = np.hstack((l, r))
    elif features == "scdv":
        matrix = np.load('../vec/fasttext_mean_scdv_x.npy', allow_pickle=True)
    elif features == "sdv":
        matrix = np.load('../vec/sdv.npy', allow_pickle=True)
    elif features == "sdv1":
        matrix = np.load('../vec/sdv1.npy', allow_pickle=True)
    elif features == "sdv2":
        matrix = np.load('../vec/sdv2.npy', allow_pickle=True)
    elif features == "bert":
        matrix = np.load('../vec/bert_x.npy', allow_pickle=True)
    elif features == "raw_text":
        df = pd.read_table("../data/train-val_pre.tsv", index_col=0)
        matrix = np.array(df["text"], dtype=str)
    return matrix

def load_y_train(features):
    if features == "bow_nva":
        return np.load('../vec/bow_train_y_nva.npy', allow_pickle = True)
    elif features == "raw_text":
        df = pd.read_table("../data/train-val_pre.tsv", index_col=0)
        return np.array(df["label"], dtype=int)
    else:
        return np.load('../vec/y_full.npy', allow_pickle=True).astype('int')

"""
"""
class Submission:

    @classmethod
    def create_submission(cls, run_name):
        submission = pd.read_csv('../input/sampleSubmission.csv')
        pred = Util.load(f'../model/pred/{run_name}-test.pkl')
        for i in range(pred.shape[1]):
            submission[f'Class_{i + 1}'] = pred[:, i]
        submission.to_csv(f'../submission/{run_name}.csv', index=False)
"""