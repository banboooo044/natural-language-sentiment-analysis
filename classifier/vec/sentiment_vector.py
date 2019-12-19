import pandas as pd
import json
import numpy as np
from sklearn.metrics import accuracy_score

def createVector(df):
    f = open("../data/sentiment-dictionary.json", "r")
    dic = json.load(f)

    for key, value in dic.items():
        if value == 0 or value == 1:
            dic[key] = 1
        else:
            dic[key] = -1
    
    x_sum = np.zeros(len(df))
    for idx, ( text, _, _ ) in df.iterrows():
        x_sum[idx] = np.sum(np.array([ dic.get(word, 0) for word in text.split(",") ]))

    return x_sum

if __name__ == "__main__":
    PATH = "../data/courpus-wakati-juman.tsv"
    df = pd.read_table(PATH, index_col=0)
    df = df[~pd.isnull(df["text"])]

    vec = createVector(df)

    y = np.array(df["label"])
    y[np.where(y != 0)] = 1
    y_pred = np.ones_like(y)
    y_pred[ np.where(vec > 0.0) ] = 0

    print(df.iloc[np.where( (y == 1) & (y_pred == 0))])
    print(accuracy_score(y, y_pred))

