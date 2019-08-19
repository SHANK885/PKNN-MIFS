# -*- coding: utf-8 -*-

# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


def KNN(dataset, features, k):


    # find shape of dataset
    #m = dataset.shape[0]
    d = dataset.shape[1] - 1

    # convertig the dataframe to numpy array
    X = dataset.iloc[:, features].values.astype(np.float64)
    y = dataset.iloc[:, d].values

     # Taking care of missing data
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X[:, 0:d-1])
    X[:, 0:d-1] = imputer.transform(X[:, 0:d-1])

    # Encoding categorical data
    # Encoding the Independent Variable
    '''
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

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Fitting K-NN to the Training set
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm_knn = confusion_matrix(y_test, y_pred)

    '''
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements
    '''
    # Accuracy
    diagonal_sum = cm_knn.trace()
    total_sum = cm_knn.sum()

    return (cm_knn, (diagonal_sum/total_sum) * 100)
