from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
import numpy as np
import random
import time

# Adapted from GitHub user ksopyla

# 'Bunch' object, similar to Dictionary with attributes
# with same name as the keys printed by the 'keys()' method
mnist = fetch_openml(name='mnist_784', data_home='./')

# print(mnist.keys())
# print(type(mnist.data))
# print(type(mnist.target))

X_data = mnist.data
Y = mnist.target

scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)
X_data, Y = list(zip(*(sorted(list(zip(X_data, Y)), key=lambda k: random.random()))))[:5000]
x_train, x_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=49)

param_c = 5
param_gamma = 0.05
classifier = svm.SVC(C=param_c, gamma=param_gamma)
# t0 = time.time()
classifier.fit(x_train, y_train)
# t1=time.time()
predicted = classifier.predict(x_test)
# score = classifier.score(x_test, y_test)
# print(score)
print(metrics.confusion_matrix(y_test, predicted))
