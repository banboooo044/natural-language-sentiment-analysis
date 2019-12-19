# 完成したシステムを使うプログラム (保留)
import xgboost as xgb
import torchtext
from torchtext.vocab import Vectors
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import sys

from text_preprocess import *
import gensim

# [平均 ,最大 , 最小値]
def create_freature(text, vocab_vec, stoi):
    ave = np.average([ vocab_vec[stoi[w]] for w in text ], axis=0)
    mx = np.max([ vocab_vec[stoi[w]] for w in text ], axis=0)
    mn = np.min([ vocab_vec[stoi[w]] for w in text ], axis=0)
    return np.r_[ ave, mx, mn ]

max_length = 25
TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing,
                            use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length)
japanese_word2vec_vectors = Vectors(name='../data/japanese_word2vec_vectors.vec')

text = "どうしよう、やってしまった。"
text_split = TEXT.preprocess(text)
vector_data = create_freature(text_split, vocab_vec=japanese_word2vec_vectors.vectors.numpy(), stoi=japanese_word2vec_vectors.stoi)
sz = len(vector_data)

model = xgb.Booster({'nthread': 4})  # init model
model.load_model('../model/01.model')  # load data

test = xgb.DMatrix(vector_data.reshape(1, sz))
result = model.predict(test)

emotions = ["happy", "sad", "disgust", "angry", "fear", "surprise"]
plt.pie(result*100)
#plt.show()
plt.savefig('../result/happy.png',dpi=300)
