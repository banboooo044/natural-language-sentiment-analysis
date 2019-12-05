## create word2vec
import pandas as pd
import logging
import numpy as np
from gensim.models import Word2Vec,KeyedVectors
from sklearn.preprocessing import normalize
import sys
import re
    
def save_model(model, filename):
    print("save model ...")
    model.save(f"./{filename}.model")

def load_model(filename):
    print("load model ... ")
    model = KeyedVectors.load_word2vec_format(f"./{filename}.model")
    return model

def createVector(df, model, num_features=200):
    print("create vector ... ")
    x = np.empty((0, num_features))
    for idx, (text, label, _) in df.iterrows():
        tmp = np.array([ np.array(model.wv[word])  for word in text.split(",") if word in model.wv.vocab ])
        if len(tmp) != 0:
            x = np.append(x, np.mean(tmp, axis=0).reshape(1, num_features), axis=0)
        else:
            x = np.append(x, np.zeros((1, num_features)), axis=0)
    return x

if __name__ == "__main__":
    f = "word2vec_pre"
    PATH = "../data/train-val-wakati-juman.tsv"
    df = pd.read_table(PATH, index_col=0)
    df = df[~pd.isnull(df["text"])]
    token = df["text"].apply(lambda x: x.split(","))
    #model = KeyedVectors.load_word2vec_format('../data/entity_vector/entity_vector.model.bin', binary=True)
    #save_model(model, f)
    model = load_model(f)
    x = createVector(df, model)
    np.save(f"./{f}_x.npy", x)