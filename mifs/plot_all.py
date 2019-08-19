# -*- coding: utf-8 -*-
"""
Created on %(date)

@author: %(shashank)
"""
import csv
import matplotlib.pyplot as plt
import numpy as np


k_value = []
acc_lr = []
acc_knn = []
acc_svm = []
acc_ksvm = []
acc_nv = []
acc_dt = []
acc_rf = []


with open("../Output/iris.csv", 'r') as csvfile:
    plots = csv.reader(csvfile)
    for rows in plots:
        k_value.append(float(rows[0]))
        acc_lr.append(float(rows[1]))
        acc_knn.append(float(rows[2]))
        acc_svm.append(float(rows[3]))
        acc_ksvm.append(float(rows[4]))
        acc_nv.append(float(rows[5]))
        acc_dt.append(float(rows[6]))
        acc_rf.append(float(rows[7]))

plt.figure(figsize = (10, 8))

'''
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
'''
plt.plot(k_value, acc_lr, 'k.-', color = 'r', label = 'Logistic Regression')
plt.plot(k_value, acc_knn, 'k.-', color = 'g', label = 'KNN')
plt.plot(k_value, acc_svm, 'k.-', color = 'b', label = 'Support Vector')
plt.plot(k_value, acc_ksvm, 'k.-', color = 'y', label = 'Kernel Support Vector')
plt.plot(k_value, acc_nv, 'k.-', color = 'm', label = 'Naive Bayes')
plt.plot(k_value, acc_dt, 'k.-', color = 'b', label = 'Decision Tree')
plt.plot(k_value, acc_rf, 'k.-', color = 'c', label = 'Random Forest')
plt.legend()
plt.title("Iris", size = 20, color = 'Blue')
plt.grid(color='g',linestyle='--', which = "both")
plt.xlabel("Number of Most Relevant Features (k)", size = 20)
plt.ylabel("Accuracy", size = 20)

plt.show()


