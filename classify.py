import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()

# ['target_names', 'feature_names', 'DESCR', 'data', 'target']
print("Keys of Iris Dataset: \n{}".format(iris_dataset.keys()))

# ['setosa' 'versicolor' 'virginica']
print("Target Names: {}".format(iris_dataset['target_names']))

# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print("Feature Names: {}".format(iris_dataset['feature_names']))

# EG: [[5.1 3.5 1.4 0.2]]
print("Data: {}".format(iris_dataset['data']))

# [0,0,0,0,1,1,1,1,1,2,2,2,2,2]
# Species of flower are encoded from 0 - 2
print("Target: {}".format(iris_dataset['target']))

# Get shape of data: (150,4) (sample, feature)
print("Shape of data: {}".format(iris_dataset['data'].shape))

# We want to split the data into both a training set, and a test set.
# 75% taken as training set, and the remaining 25% used as a test set.
# Data is denoted by X and labels denoted by y
# random_state is used to generate arandom fixed seed so the output is deterministic
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))

print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))

print("y_test shape: {}".format(y_test.shape))

# Create a scatter plot to visualize pairs of data
# This is done to see if we have any outliers

# Label columns using our feature names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# Create scatter matrix from the DataFrame, color by y_train
# pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)

# Instantiate classifier
knn = KNeighborsClassifier(n_neighbors=1)

# Train model
knn.fit(X_train, y_train)

print("Model: {}".format(knn))

# Let's try some predictions
# SKLearn expects a 2-diemensional array for data
X_new = np.array([[5, 2.9, 1.3, 0.2]])

prediction = knn.predict(X_new)

print("Prediction: {}".format(prediction))
print("Predicted Target: {}".format(iris_dataset['target_names'][prediction]))

# Evaluate our model
y_pred = knn.predict(X_test)

print("Test set prediction: {}".format(y_pred))

# Check score
print("Test Score: {}".format(knn.score(X_test, y_test)))
