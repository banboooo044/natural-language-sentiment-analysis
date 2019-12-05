import pandas as pd
import numpy as np

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


def train(df, dm=0, min_count=1):
    print("train model ...")
    sentences = [ TaggedDocument(words=words.split(","), tags=[i]) for i, (words, _, _) in df.iterrows() ]
    model = Doc2Vec(documents=sentences, vocab_size=200,alpha=0.0025,min_alpha=0.000001, \
                    window=15, iter=80, min_count=min_count, dm=dm, workers=4)
    print('\n訓練開始')
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    return model

def save_model(model, filename):
    print("save model ...")
    model.save(f"./{filename}.model")

def load_model(filename):
    print("load model ... ")
    model = Doc2Vec.load(f"./{filename}.model")
    return model

def createVector(df, model):
    print("create vector ... ")
    return np.array([ np.array(model.docvecs[idx]) for idx, _ in df.iterrows() ] )

if __name__ == "__main__":
    f = "doc2vec"
    PATH = "../data/train-val-wakati-juman.tsv"
    df = pd.read_table(PATH, index_col=0)
    df = df[~pd.isnull(df["text"])]
    model = train(df)
    x = createVector(df, model)
    np.save(f"./{f}_x.npy", x)

