# -*- coding: utf-8 -*-

import load_dataset
import preprocess_data
from models.LR import logisticRegression
from models.KNN import KNN
from models.SVM import supportVector
from models.KernelSVM import kSVM
from models.NaiveBayes import naiveBayes
from models.DecisionTree import decisionTree
from models.RandomForest import randomForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load DataSets  d = number of features  F = {Set of Features}
dataset_name = input("Please Enter Dataset Name:")

data, label, noObjects, noFeatures, dataset = load_dataset.loadData(dataset_name)

print("Number of Objects: ", noObjects)
print("Number of Features: ", noFeatures)

# Preprocessing Data----- dataPro: Processed Features, labelPro: Processed Label
# shape(dataPro) = (noObjects, noFeatures)
# shape(labelPro) = (noObjects, 1)
dataPro, labelPro = preprocess_data.dataPreprocessing(data, label, noFeatures)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(dataPro,
                                                    labelPro,
                                                    test_size = 0.25,
                                                    random_state = 0)
print("Number of Test Instance: ", len(X_test), "%")

print("")
print("--------------------------------------------------------------------------")
print("                    Logistic Regression Classification                    ")
print("")

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_lr = classifier.predict(X_test)

# Making the Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Accuracy
diagonal_sum_lr = cm_lr.trace()
total_sum_lr = cm_lr.sum()

accuracy_lr = round((diagonal_sum_lr/total_sum_lr) * 100, 2)
print("Accuracy with Logistic regression models is: ", accuracy_lr)

print("")
print("--------------------------------------------------------------------------")
print("                    Naive Bayes Classification                    ")
print("")





