# -*- coding: utf-8 -*-

import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

#y = np.random.randint(low=0, high=10, size=100)
y = [1,0,2,1,0,1,0,2,2,1,1,0,2,0,1]
y_p = [1,0,2,0,0,1,0,2,0,1,1,0,2,0,1]
#y_p = np.random.randint(low=0, high=10, size=100)

def plot_classification_report(y_tru, y_prd, figsize=(6, 6), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = list(np.unique(y_tru))
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sn.heatmap(rep,
                annot=True,
                cbar=False,
                xticklabels=xticks,
                yticklabels=yticks,
                ax=ax)

plot_classification_report(y, y_p)