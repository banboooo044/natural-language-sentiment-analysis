# T-SNEを用いた文章分散表現の可視化
import sys, os
sys.path.append('../')

from src import *

import matplotlib.pyplot as plt
import  japanize_matplotlib

import numpy as np
from scipy import sparse

def visualize(xs, ys, labels, dataname):
    plt.figure(figsize=(15, 15), dpi=300)
    emotion = [ "happy", "sad", "angry", "fear/surprise", "other" ]
    np.savez_compressed(f"./fig/{dataname}.npz", xs, ys, labels)
    for i, e in enumerate(emotion):
        idx = np.where(labels==i)
        plt.scatter(xs[idx], ys[idx], s=200, label=e)
    plt.legend()
    plt.title(dataname)
    plt.savefig(f'./fig/{dataname}.png', dpi=300)

def train(vec):
    model = BHTSNE(n_components=2, random_state=71)
    np.set_printoptions(suppress=True)
    return model.fit_transform(vec)

x = np.load("./word2vec_fine-tuning-iter5_x.npy", allow_pickle=True)
y = np.load("./y_full.npy", allow_pickle=True)
dataname = "word2vec_pre_all"
xy = train(x)
np.random.seed(seed=71)
idx = np.array([ np.random.choice(np.where(y == i)[0], 120) for i in range(5) ] ).reshape(5, -1).flatten()
xs,ys,labels = xy[idx, 0], xy[idx, 1], y[idx]
visualize(xs, ys, labels, dataname)