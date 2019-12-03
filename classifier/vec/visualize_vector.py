from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import  japanize_matplotlib

import numpy as np

def visualize(xs, ys, labels, filename):
    plt.figure(figsize=(15, 15), dpi=300)
    emotion = [ "happy", "sad", "angry", "fear/surprise", "other" ]
    for i, e in enumerate(emotion):
        idx = np.where(labels==i)
        plt.scatter(xs[idx], ys[idx], s=200, label=e)

    plt.legend()
    plt.title(filename)
    plt.savefig(f'./fig/{filename}.png', dpi=300)

def train(vec):
    model = TSNE(n_components=2, random_state=71, verbose=2)
    np.set_printoptions(suppress=True)
    model.fit_transform(vec)
    return model.embedding_

x = np.load("./word2vec_mean_x.npy", allow_pickle=True)
y = np.load("./word2vec_mean_y.npy", allow_pickle=True)
xy = train(x)
np.random.seed(seed=71)
idx = np.array([ np.random.choice(np.where(y == i)[0], 40) for i in range(5) ] ).reshape(5, 40).flatten()
xs,ys,labels = xy[idx, 0], xy[idx, 1], y[idx]
visualize(xs, ys, labels, "word2vec_mean")
