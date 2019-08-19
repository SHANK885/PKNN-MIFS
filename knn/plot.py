# -*- coding: utf-8 -*-
"""
Created on %(date)

@author: %(shashank)
"""
import csv
import matplotlib.pyplot as plt
import numpy as np


k_value = []
acc = []

with open("../Output/wine--k-vs-accuracy.csv", 'r') as csvfile:
    plots = csv.reader(csvfile)
    for rows in plots:
        k_value.append(float(rows[0]))
        acc.append(float(rows[1]))

plt.figure(figsize = (10, 8))


def annot_max(k_value, acc, ax = None):
    xmax = k_value[np.argmax(acc)]
    ymax = max(acc)

    text = "k={:0.2f}, acc={:0.2f}".format(xmax, ymax)

    if not ax:
        ax = plt.gca()

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="bottom")
    ax.annotate(text, xy=(xmax,ymax), xytext=(0.94,0.96), **kw, color='r')


annot_max(k_value, acc)
plt.plot(k_value, acc, 'k.-', color = 'm', label = "KNN")
plt.legend(loc = 'lower right')
plt.title("Wine", size = 20, color = 'Blue')
plt.grid(color='g',linestyle='--', which = "both")
plt.xlabel("Number of Neighbors (k)", size = 20)
plt.ylabel("Accuracy", size = 20)

plt.show()


