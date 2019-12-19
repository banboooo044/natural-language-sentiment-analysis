# Word2vec, Fasttext の単語分散表現から文章分散表現を取得

## 学習済みモデルをそのまま使う場合:
## word2vec : python3 get_embedding_matrix word2vec
## fasttext : python3 get_embedding_matrix fasttext
## fasttext+neologd : python3 get_embedding_matrix fasttext-neologd

## 自分で学習した [filename].model を使う場合
## python3 get_embedding_matrix filename
## modelファイルの形式によっては, 関数load_model をいじらないとダメかもしれない.

import sys, re
import numpy as np
import pandas as pd
import logging
from gensim.models import Word2Vec,KeyedVectors
from sklearn.preprocessing import normalize

""" 
This is a model which is described in "Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms"
See  for details https://arxiv.org/pdf/1805.09843.pdf
"""

# SWEM - hier
def hier(vecs, n=2):
    sz, k = vecs.shape
    ret = np.ones((1, k)) * (-np.inf)
    if sz == 1:
        vecs = np.vstack((vecs, np.zeros((2, k))))
    elif sz == 2:
        vecs = np.vstack((vecs, np.zeros(k)))
    sz, k = vecs.shape
    for i in range(sz - n + 1):
        ret = np.max(np.vstack((ret, np.mean(vecs[i:i+n], axis=0))), axis=0)
    return ret

def load_model(path):
    model = KeyedVectors.load(f'./{path}.model')
    return model
    
# ベクトルを作る
def createVector(df, model, num_features):
    print("create vector ... ")
    x_mean = np.zeros((len(df), num_features))
    x_max = np.zeros((len(df), num_features))
    x_hier = np.zeros((len(df), num_features))
    for idx, (text, label, _) in df.iterrows():
        tmp = np.array([ np.array(model.wv[word]) if word in model.wv.vocab else np.random.randn(num_features) for word in text.split(",")  ])
        if len(tmp) != 0:
            # SWEM-aver
            x_mean[idx] = np.mean(tmp, axis=0).reshape(1, num_features)
            # SWEM-max
            x_max[idx] = np.max(tmp, axis=0).reshape(1, num_features)
            # SWEM-hier
            x_hier[idx] = hier(tmp, n=2)
    return x_mean, x_max, x_hier

if __name__ == "__main__":
    PATH = "../data/corpus-wakati-juman.tsv"
    df = pd.read_table(PATH, index_col=0)
    df = df[~pd.isnull(df["text"])]
    token = df["text"].apply(lambda x: x.split(","))

    args = sys.argv
    # Word2vec
    if args[1] == "word2vec":
        model = KeyedVectors.load_word2vec_format('./word2vec_pre/entity_vector.model.bin', binary=True)
        OUTPUT_FILENAME = "word2vec"
    # Fasttext
    elif args[1] == "fasttext":
        model = KeyedVectors.load_word2vec_format('./fasttext_pre/fasttext.vec', binary=False)
        OUTPUT_FILENAME = "fasttext"
    # Fasttext + neologd辞書
    elif args[1] == "fasttext-neologd":
        model = KeyedVectors.load_word2vec_format('./fasttext_pre/fasttext-neologd.vec', binary=False)
        OUTPUT_FILENAME = "fasttext-neologd"
    # その他
    else:
        model = load_model(args[1])
        OUTPUT_FILENAME = input("[filename].model")

    x_mean, x_max, x_hier = createVector(df, model, num_features=model.vector_size)

    np.save(f"./{OUTPUT_FILENAME}_aver.npy", x_mean)
    np.save(f"./{OUTPUT_FILENAME}_max.npy", x_max)
    np.save(f"./{OUTPUT_FILENAME}_hier.npy", x_hier)