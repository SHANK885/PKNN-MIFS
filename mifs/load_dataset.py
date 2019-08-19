# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# call loadData function
def loadData(dataset_name):

    # Load the dataset
    path = "datasets/" + dataset_name + ".csv"
    dataset = pd.read_csv(path, header = None)
    # data = dataset.iloc[:, :].values

    # Compute Shape of Dataset
    m = dataset.shape[0]
    d = dataset.shape[1] - 1

    # Filter Out Features and Labels
    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y = dataset.iloc[:, d].values

    '''
    for i in range(0,3):
        print("")
    print("DataSet ", dataset_name + ".csv Loaded----------------------"  )
    print("")
    '''
    # Return Features, Labels, Number of Objects & Number of Features
    return (X, y, m, d, dataset)







