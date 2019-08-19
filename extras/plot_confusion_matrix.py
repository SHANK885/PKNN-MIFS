# -*- coding: utf-8 -*-
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[228,   0,   0,   1,   4,  27,   4,   3,   0,   1,   0,   0],
 [  0,   1,   0,   0,   0,  0,   0,   0,   0  , 0,   0,   0],
 [  0,   0,   1,   0,   1,   0,   0,   0,   0  , 0 ,  0,   0],
 [  4,   0,   1,  15,   5,   3,   0,   1,   0  , 0  , 1,   0],
 [  6,   0,   0,   0, 38,   6 ,  1 ,  0 ,  0  , 0,   1,   0],
 [ 32,   0,   0,   1,   2, 81 ,  5 ,  1 ,  0  , 0 ,  0,   0],
 [  6,   0,   0,   0,   0,  2 , 20 ,  1 ,  0 ,  0  , 0,   0],
 [  9,   0,   0,   0,   6,  0 ,  1 , 15 ,  0   ,0  , 0,   0],
 [  1,   0,   0,   0,   0,   1,   0,   0,  38  , 0  , 0,   1],
 [  1,   0,   0,   0,   0,   0,   0,   0,   0 ,  2  , 0 ,  0],
 [  1,   0,   1,   0,   1,   0,   0,   0,   0,  0  ,31 ,  0],
 [  2,   0,   0,   0,   1,   0,   0 ,  0,  0,   0,   0,  11]]


df_cm = pd.DataFrame(array, range(len(array)), range(len(array)))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
