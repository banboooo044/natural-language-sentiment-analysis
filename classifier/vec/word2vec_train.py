# train Word2Vec and save model
import sys, re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

class EpochLogger(CallbackAny2Vec):
    def __init__(self, path_prefix):
        self.epoch = 0
    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

# skip-gram
def train(token):
    print("train model ...")
    epoch_logger = EpochLogger()
    ## sg: 1 -> skip-gram, 0 -> CBOW
    model = Word2Vec(token, workers=40, hs=0, sg=1, negative = 10, iter = 25,\
            size=200, min_count=10, \
            window=10, sample =1e-3, seed=1, compute_loss=True, callbacks=[epoch_logger])
    return model
    
def save_model(model, filename):
    print("save model ...")
    model.save(f"./{filename}.model")

if __name__ == "__main__":
    PATH = "../data/courpus-wakati-juman.tsv"
    df = pd.read_table(PATH, index_col=0)
    df = df[~pd.isnull(df["text"])]
    token = df["text"].apply(lambda x: x.split(","))

    model = train(token)
    # モデルの保存
    OUTPUT_FILENAME = "word2vec"
    save_model(model, OUTPUT_FILENAME)