# -*- coding: utf-8 -*-

from mifs import mifs
import numpy as np
#from mifs.models.KNN import KNN
from knn import knn_final
import csv


if __name__ == "__main__":
    
    dataset_name, selectedFeatures, noFeatures, dataset, k = mifs.mifs_nd()
    
    print("")
    print("Selected Features: ", selectedFeatures)
    
    m = [f for f in range(1, k+1)]
    
    allFeatures = []
    for item in m:
        allFeatures.append(selectedFeatures[0:item])
    
    allFeatures.append([ _ for _ in range(0, noFeatures)])
    
    
    print("")
    print("--------------------------------------------------------------------------")
    print("                           KNN Classification                             ")
    print("")
    
    
    # for paralle computation put parallel = True
    
    k = 5
    features = allFeatures[-1]
    feature_length = len(features)
    k, accuracy = knn_final.my_knn(dataset_name,
                                   features,
                                   dataset,
                                   k,
                                   feature_length,
                                   parallel = False
                                   )
   
# code for accuracy analysis for various values of k and FL
    '''
    for k in range(1, 16, 2):
    
        print("No. of Features vs Accuracy - K  Fixed")
    
        values = []
        values.append(["No. Best Features", "Accuracy"])
        i = 0
        acc = 0
        for features in allFeatures:
    
            feature_length = len(features)
            k, accuracy = knn_final.my_knn(dataset_name, features, dataset, k, feature_length, parallel = False)
            values.append([len(features), accuracy])
            print("Accuracy with best ", feature_length, "features is: ", accuracy, " %")
    
            acc += accuracy
            i += 1
    
        print("Average Accuracy for ", k, " Neighbors is: ", round((acc/i),2), "%")
        #store feature vs accuracy K fixed
        save_ftr_vs_accuracy(feature_length, k, dataset_name, values)
    
    
    for features in allFeatures:
    
        print("K vs Accuracy - Length of Features Fixed")
    
        values = []
        values.append(["No Neighbors", "Accuracy"])
        i = 0
        acc = 0
        for k in range(1, 16, 2):
    
            feature_length = len(features)
            k, accuracy = my_knn(dataset_name, features, dataset, k, feature_length, parallel = False)
            values.append([k, accuracy])
            print("Accuracy with best ", feature_length, "features is: ", accuracy, " %")
    
            acc += accuracy
            i += 1
    
        print("Average Accuracy for ", feature_length, " Best Feature is: ", round((acc/i),2), "%")
        # store k vs accuracy FL fixed
        save_k_vs_accuracy(feature_length, k, dataset_name, values)
    
    '''
  
    
def save_ftr_vs_accuracy(fl, k, dataset_name, values):

    filename = dataset_name + "-" +str(k) + "-neighbors-" + "-bstftr-vs-accuracy.csv"
    filename = "/Outputs/"+filename
    values = np.array(values)
    with open(filename,"w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(values)
    my_csv.close()

    return


def save_k_vs_accuracy(fl, k, dataset_name, values):

    filename = dataset_name + "-best-" + str(fl) + "-features-k-vs-accuracy.csv"
    filename = "/Outputs/"+filename
    values = np.array(values)
    with open(filename,"w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(values)
    my_csv.close()

    return
    
