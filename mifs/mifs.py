# -*- coding: utf-8 -*-

# Import Necessary Packages
#import numpy as np
#import pandas as pd
from mifs import load_dataset
from mifs import preprocess_data
from mifs import feature_class_correlation
from mifs import feature_feature_correlation
from mifs import domination_count
import time




def mifs_nd():

    # Load DataSets  d = number of features  F = {Set of Features}
    dataset_name = input("Please Enter Dataset Name:")
    data, label, noObjects, noFeatures, dataset = load_dataset.loadData(dataset_name)

    print("Number of Objects: ", noObjects)
    print("Number of Features: ", noFeatures)


    k = int(input("Input Number of Feature You Want to Select: " ))
    print("")

    while k > noFeatures:
        print("K: ", k, "is Greater Than Number of Features: ", noFeatures)
        print("Please Re Enter Value of K")
        k = int(input("K = "))


    assert(k <= noFeatures)

    # Preprocessing Data----- dataPro: Processed Features, labelPro: Processed Label
    # shape(dataPro) = (noObjects, noFeatures)
    # shape(labelPro) = (noObjects, 1)
    dataPro, labelPro = preprocess_data.dataPreprocessing(data, label, noFeatures)

    # Start Time
    #tart = time.time()

    # for i in range(1, d+1): Compute Feature Class Mutual Information  MI(fi, C)
    # shape(featureClassCorr) = (noFeatures, 1)
    featureClassCorr = feature_class_correlation.featureClassCorrelation(dataPro,
                                                                         labelPro,
                                                                         noObjects,
                                                                         noFeatures)

    #stop_FCMI = time.time()

    # for i in range(1, d+1): Compute Feature Feature Mutual Information  MI(fi, fj)
    # shape(featureFeatureCorr) = (noFeatures, noFeatures)
    featureFeatureCorr = feature_feature_correlation.featureFeatureCorrelation(dataPro,
                                                                               noObjects,
                                                                               noFeatures)

    #stop_FFMI = time.time()

    # Define and Initialize Selected and Left Feature List
    selectedFeature = []
    leftFeature = [i for i in range(0, noFeatures)]

    '''
    max_val = max(l)
    max_idx = l.index(max_val)
    return max_idx, max_val
    '''
    print("-----------------------Starting Feature Selection---------------------")
    print("")

    # Find Value of  Maximum Feature-Class Mutual Information
    max_val = max(list(featureClassCorr))
    #print("max_val: ", max_val)

    # # Find Index of Maximum Feature-Class Mutual Information
    max_idx = list(featureClassCorr).index(max_val)
    #print("max_idx: ", max_idx)

    # Select the feature fi "selectedFeature" with maximum Feature-Class Information
    selectedFeature.append(max_idx)

    #print("First Selected Feature: ", selectedFeature)


    # Delete the Selected Feature From the Left Feature List
    leftFeature.remove(max_idx)

    #print("Left Features: ", leftFeature)

    print()
    print("----------------------------------------------------------------------")
    print("")

    #start_loop = time.time()
    # Select K-1 More Features
    iter = 0
    while len(selectedFeature) < k:

        iter = iter + 1
        print("------------------------  Iteration :", iter, "  -------------------------------" )

        #tic = time.time()

        # Compute Domination Count For Left Features
        dominationCount, dominatedCount = domination_count.computeDominationCount(selectedFeature,
                                                                                 leftFeature,
                                                                                 featureFeatureCorr,
                                                                                 featureClassCorr)
        #tac = time.time()

        # countDiff Vector : (Cd -Fd) ie (dominationCount - dominatedCount)
        countDiff = (dominationCount - dominatedCount)[0]
        #print("CountDiff (Cd - Fd): ", countDiff)
        #print("")

        # Find max((dominationCount - dominatedCount))
        max_count_diff = max(countDiff.T)
        #print("max_count_diff: ", max_count_diff)
        #print("")

        # Find Index of max((dominationCount - dominatedCount))
        max_count_idx = list(countDiff.T).index(max_count_diff)
        #print("max_count_idx: ", max_count_idx)
        #print("")

        max_val = featureClassCorr[leftFeature[max_count_idx]]
        #print("max_val: ", max_val)
        #print("")

        # Index to feature to Maximum FCMI and Minimum FFMI
        max_idx = list(featureClassCorr).index(max_val)

        #print("Selected Feature Index: ", max_idx)
        #print("")

        # Select the feature with maximum FCMI and Minimum FFMI
        selectedFeature.append(max_idx)

        # Find Number of Selected Feature Till Now
        #print("Number of Selected Feature :", len(selectedFeature))
        #print("Selected Feature: ", selectedFeature)
        #print("")

        #print("max_idx:", max_idx)
        # Delete the Selected Feature From the Left Feature List
        leftFeature.remove(max_idx)

        # Find Number of Left Feature Till Now
        #print("Number of Left Feature :", len(leftFeature))
        #print("Left Feature: ", leftFeature)

        #print("")
        #print("")


    # Stop Time
    #stop = time.time()

    print("-------------------- ", k, "Features Selected ----------------------")
    print("Final Selected Feature: ", selectedFeature)
    print("")
    print("Final Left Feature: ", leftFeature)
    print("")

    '''
    print("Time Taken For Computing Feature Class Correlation: "
          + str(1000 * (stop_FCMI - start)) + " ms")
    print("")

    print("Time Taken For Computing Feature Feature Correlation: "
          + str(1000 * (stop_FFMI - stop_FCMI)) + " ms")
    print("")

    print("Time Taken for Computing Domiation Count (1 time): ", str( 1000 * (tac - tic)), " ms")
    print("")

    print("Time Taken in the For Loop: " + str(1000 * (stop - start_loop)) + " ms")
    print("")

    print("Total Time Taken: " + str(1000 * (stop - start)) + " ms")
    '''

    return(dataset_name, selectedFeature, noFeatures, dataset, k)





#mifs_nd()