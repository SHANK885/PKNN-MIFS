# -*- coding: utf-8 -*-

from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

def dataPreprocessing(X, y, d):
    
    # Taking care of missing data
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X[:, 0:d-1])
    X[:, 1:d] = imputer.transform(X[:, 1:d])
    
    
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
    '''
    X_test = sc.transform(X_test)
    '''

    print("Loaded Data Preprocessed for Feature Selection------------------------")
    print("")
    
    return (X, y)
    
    