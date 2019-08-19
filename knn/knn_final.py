# -*- coding: utf-8 -*-
"""
Created on %(08/05/2019)

@author: %(shashank shekhar)
"""

import csv
import time
import operator
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
from knn.distances import euclidean_distances
#import sys
'''
# load dataset
def loadDataset(filename):

    # Load the dataset
    path = "/datasets/" + filename + ".csv"
    dataset = pd.read_csv(path, header = None)

    # replace empty space with NaN
    dataset[:] = dataset[:].replace("na", np.NaN)

    # Compute Shape of Dataset
    m = dataset.shape[0]
    d = dataset.shape[1] - 1

    # Filter Out Features and Labels
    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y = dataset.iloc[:, d].values

    print("\nDataSet: ", filename)

    return(X, y, m, d)
'''
# preprocess data
def preprocessData(X, y, d):

    # Taking care of missing data
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X[:, 0:d])
    X[:, 0:d] = imputer.transform(X[:, 0:d])

    # Encoding categorical data
    # Encoding the Independent Variable
    '''
    from sklearn.preprocessing import OneHotEncoder
    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

    onehotencoder = OneHotEncoder(categorical_features = [0])
    X = onehotencoder.fit_transform(X).toarray()
    '''
    # Encoding the Dependent Variable
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    # Feature Scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)

    return(X, y)

# split into train and test set
def splitDataset(X, y, test_size):

    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size = test_size, random_state = 0)

    y_trn = y_trn.reshape(len(y_trn), 1)
    y_tst = y_tst.reshape(len(y_tst), 1)

    trainingSet = np.concatenate((X_trn, y_trn), axis = 1)
    testSet = np.concatenate((X_tst, y_tst), axis = 1)

    return(trainingSet, testSet)


#############################
# SIMILARITY CHECK FUNCTION #
#############################

# euclidean distance calculation
def euclideanDistanceSerial(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return np.sqrt(distance)



def euclideanDistanceParallel(test_instance, train_set, length):

    # reshape data
    test_instance = test_instance.reshape(1, len(test_instance))

    #convert to float32
    test_instance = test_instance.astype(np.float32)
    train_set = train_set.astype(np.float32)

    # strip off the labels
    test_instance = test_instance[:, 0:length]
    train_set = train_set[:, 0:length]

    distance = euclidean_distances(train_set, test_instance, inverse=False)

    return(distance)

############################################################
# NEIGHBOURS - selecting subset with the smallest distance #
############################################################

def getNeighbors(trainingSet, testInstance, k, parallel):

    distances = []
    length = len(testInstance) - 1

    if parallel == True:
        '''
        dist = euclideanDistanceParallel(testInstance, trainingSet, length)
        dist = dist.squeeze()
        for x in range(len(trainingSet)):
            distances.append((trainingSet[x], dist[x]))
        '''
    else:
        for x in range(len(trainingSet)):
            dist = euclideanDistanceSerial(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


######################
# PREDICTED RESPONSE #
######################

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

######################
# MEASURING ACCURACY #
######################

def getAccuracy(testSet, predictions):

    cm_knn = confusion_matrix(testSet[:, [-1]], predictions)

    # Accuracy
    diagonal_sum = cm_knn.trace()
    total_sum = cm_knn.sum()

    return ((diagonal_sum/total_sum) * 100, cm_knn)



def draw_cm(dataset_name, k, array, fl):

    df_cm = pd.DataFrame(array, range(len(array)), range(len(array)))
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)
    ax = plt.axes()
    sn.heatmap(df_cm,
               linewidths = 0.3,
               annot = True,
               ax=ax,
               annot_kws = {"size": 16})

    name = "Confusion Matrix [ "+dataset_name+" ]"+" [ K:"+str(k)+"]"+" [ FL:"+str(fl)+" ]"
    ax.set_title(name)
    plt.show()

    return


def plot_classification_report(k, dataset_name, y_tru, y_prd, fl, ax = None):

    #k, dataset_name, testLabels, predictions
    #ax=None
    plt.figure(figsize=(10,7))

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = list(np.unique(y_tru))
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)
    ax = plt.axes()

    sn.heatmap(rep,
                annot=True,
                linewidths = 0.3,
                xticklabels=xticks,
                yticklabels=yticks,
                ax=ax)

    name = "Classification Report [ "+dataset_name+" ]"+"[ K:"+str(k)+" ]"+"[ FL:"+str(fl)+" ]"
    ax.set_title(name)
    plt.show()



def my_knn(dataset_name, feature_idx, dataset, k, feature_length, parallel = False):

    # Compute Shape of Dataset
    no_instances = dataset.shape[0]
    no_features = len(feature_idx)

    # Filter Out Features and Labels
    features = dataset.iloc[:, :-1].values.astype(np.float32)
    features = features[:, feature_idx]
    labels = dataset.iloc[:, [-1]].values

    # preprocess data
    X_pro, y_pro = preprocessData(features, labels, no_features)

    print("------------------------KNN Classification------------------------\n\n")

    print("Number of Instances: ", no_instances)
    print("Number of Features: ", no_features)

    if(no_instances <= 20000):
        testFraction = 0.2
    elif(no_instances > 20000 and no_instances <= 100000):
        testFraction = 0.1
    elif(no_instances > 100000 and no_instances <= 1000000):
        testFraction = 0.01
    elif(no_instances > 1000000 and no_instances <= 5000000):
        testFraction = .005
    elif(no_instances > 5000000):
        testFraction = 0.001

    # split the data into train and test set
    trainingSet, testSet = splitDataset(X_pro, y_pro, testFraction)

    testLabels = testSet[:, [-1]]

    print("Length of Test Set: ", len(testSet))
    print("\n")

    if parallel:
        print("-----------Parallel Mode------------\n")
    else:
        print("-----------Serial Mode------------\n")

    # start time
    start = time.time()

    # generate predictions
    predictions = []

    len_test_set = len(testSet)

    for i in range(len_test_set):
        neighbors = getNeighbors(trainingSet, testSet[i], k, parallel)
        result = getResponse(neighbors)
        predictions.append(result)

    accuracy, confusionMatrix = getAccuracy(testSet, predictions)

    print("")
    print("K: ", k)
    print('Accuracy: ' , round(accuracy,2) ,'%')
    print("")
    print("[CONFUSION MATRIX]\n")
    print(confusionMatrix)
    print("")

    stop = time.time()

    draw_cm(dataset_name, k, confusionMatrix, feature_length)
    plot_classification_report(k, dataset_name, testLabels, predictions, feature_length)

    if parallel == True:
        print("\nTime for Parallel Computation = ", str(1000 * (stop - start)) + " ms")
        print("")
    else:
        print("\nTime for Serial Computation = ", str(1000 * (stop - start)) + " ms")
        print("")



    return(k, round(accuracy, 2))
