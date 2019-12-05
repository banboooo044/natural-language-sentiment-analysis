import sys, os
sys.path.append('../')

import numpy as np
import plotly
import plotly.graph_objs as go

from src.util import Util

plotly.offline.init_notebook_mode()

def visualize(xs, ys, labels):
    trace = []
    emotion = [ "happy", "sad", "angry", "fear/surprise", "other" ]
    for i, e in enumerate(emotion):
        idx = np.where(labels==i)
        t = go.Scatter(x=xs[idx], y=ys[idx], mode='markers', name=e)
        trace.append(t)

    fig = dict(data = trace)
    plotly.offline.iplot(fig)

arrays = np.load("./fig/word2vec_mean.npz", allow_pickle=True)
xs, ys, labels = arrays['arr_0'], arrays['arr_1'],arrays['arr_2']
visualize(xs, ys, labels)
    


