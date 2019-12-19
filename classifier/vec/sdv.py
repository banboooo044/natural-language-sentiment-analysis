# 極性辞書を用いた文章分散表現を獲得
import pandas as pd
import numpy as np
import json
from gensim.models import Word2Vec, TfidfModel, KeyedVectors
from gensim.corpora import Dictionary

from tqdm import tqdm
import warnings

warnings.simplefilter('ignore', DeprecationWarning)

def createVector(df, model):
    print("create Vector")
    # Sentiment Dictionary を用いる
    f = open("../data/sentiment-dictionary.json", "r")
    dic = json.load(f)
    for key, value in dic.items():
        if value == 0 or value == 1:
            dic[key] = 1
        else:
            dic[key] = -1
    num_features = model.vector_size
    sdv = np.zeros((len(df), num_features*3))
    for idx, (text, label, _) in tqdm(df.iterrows()):
        neu = np.zeros((1, num_features))
        pos = np.zeros((1, num_features))
        neg = np.zeros((1, num_features))
        neu_num, pos_num, neg_num = 0,0,0
        for word in text.split(","):
            if word in dic:
                if dic[word] == 1:
                    if word in model.wv.vocab:
                        pos_num += 1
                        pos = pos + np.array(model.wv[word])
                else:
                    if word in model.wv.vocab:
                        neg_num += 1
                        neg = neg + np.array(model.wv[word])
            else:
                if word in model.wv.vocab:
                    neu_num += 1
                    neu = neu + np.array(model.wv[word])
        neu_num = max(neu_num, 1)
        pos_num = max(pos_num,1)
        neg_num = max(neg_num, 1)
        sdv[idx] = np.hstack((neu/neu_num, pos/pos_num, neg/neg_num))
    return sdv

if __name__ == "__main__":
    PATH = "../data/corpus-wakati-juman.tsv"
    df = pd.read_table(PATH, index_col=0)
    df = df[~pd.isnull(df["text"])]
    
    model = KeyedVectors.load_word2vec_format('./fasttext.bin', binary=True)
    document = list(df["text"].apply(lambda x: x.split(',')))

    vec = createVector(df, model)
    np.save('./sdv.npy', vec)