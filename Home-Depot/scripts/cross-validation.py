# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 23:33:52 2016

@author: abhishek
"""
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

np.random.seed(1729)

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

corpus_svd = np.loadtxt('./data/train_text_svd.txt')
corpus_test_svd = np.loadtxt('./data/test_text_svd.txt')

X_train, X_test, y_train, y_test = train_test_split(corpus_svd, train.relevance, test_size=0.3,
                                                    random_state=44)


scaler = StandardScaler()
svr = SVR()

pipeline = Pipeline([('scaler', scaler), ('svr', svr)])
pipeline.fit(X_train, y_train)

predsTrain = pipeline.predict(X_train)
predsTest = pipeline.predict(X_test)

print 'RMSE on training examples %f ' %(np.sqrt(mean_squared_error(y_train, predsTrain)))
print 'RMSE on test examples %f ' %(np.sqrt(mean_squared_error(y_test, predsTest)))
