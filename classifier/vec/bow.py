# 単語文章行列, n-gram行列

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz

# 分かち書きされたデータを読み込む
PATH = "../data/corpus-wakati-juman.tsv"
df = pd.read_table(PATH, index_col=0)
df = df[~pd.isnull(df["text"])]

# 単語文章行列
cv = CountVectorizer()
matrix = cv.fit_transform(df["text"])

# n-gram(n = 1,2,3) をvocabraryとする行列
cv_n = CountVectorizer(ngram_range=(1,3), min_df=1)
matrix_n = cv_n.fit_transform(df["text"])

# 保存
save_npz('../vec/bow_train.npz', matrix)
save_npz('../vec/n-gram.npz', matrix_n)