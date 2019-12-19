# tf-idf, tf-idf + n-gram行列

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# 分かち書きされたデータを読み込む
PATH = "../data/corpus-wakati-juman.tsv"
df = pd.read_table(PATH, index_col=0)
df = df[~pd.isnull(df["text"])]

# tf-idf
cv = TfidfVectorizer()
matrix =  cv.fit_transform(df["text"])

# tf-idf + n-gram
cv_n = TfidfVectorizer(ngram_range=(1,3), min_df=2)
matrix_n = cv_n.fit_transform(df["text"])

# 保存
save_npz('../vec/tf-idf_x.npz', matrix)
save_npz('../vec/n-gram-tf-idf_x.npz', matrix_n)
