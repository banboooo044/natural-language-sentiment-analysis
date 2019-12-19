# train Doc2vec and save model
import pandas as pd
import numpy as np

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


# PV-DBOW
def train_DBOW(df):
    print("train model ...")
    sentences = [ TaggedDocument(words=words.split(","), tags=[i]) for i, (words, _, _) in df.iterrows() ]
    model = Doc2Vec(documents=sentences, vector_size=300, alpha=0.025, min_alpha=0.000001, \
                    window=15, iter=20, min_count=1, dm=0, workers=4, seed=71)
    
    print('train start')
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    return model

# PV-DM
def train_DM(df):
    print("train model ...")
    sentences = [ TaggedDocument(words=words.split(","), tags=[i]) for i, (words, _, _) in df.iterrows() ]
    model = Doc2Vec(documents=sentences, vector_size=300, alpha=0.05, min_alpha=0.000001, \
                    window=10, iter=20, min_count=1, dm=1, workers=4, seed=71)
    
    print('train start')
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    return model  


def save_model(model, filename):
    print("save model ...")
    model.save(f"./{filename}.model")

if __name__ == "__main__":
    PATH = "../data/courpus-wakati-juman.tsv"
    df = pd.read_table(PATH, index_col=0)
    df = df[~pd.isnull(df["text"])]

    model = train_DM(df)

    # モデルの保存
    OUTPUT_FILENAME = "Doc2vec"
    save_model(model, OUTPUT_FILENAME)