# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 22:33:15 2016

@author: abhishek
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

np.random.seed(44)

train = pd.read_csv('./data/train_processed.csv')
test = pd.read_csv('./data/test_processed.csv')

X = train[train.columns.drop('TARGET')]
y = train.TARGET

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1279)

scaler = MinMaxScaler()
clf = LogisticRegression()
clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, class_weight='auto')
clf = XGBClassifier()

pipeline = Pipeline([('scaler', scaler), ('clf', clf)])
pipeline.fit(X_train, y_train)


predsTrain = pipeline.predict_proba(X_train)[:, 1]
predsTest = pipeline.predict_proba(X_test)[:, 1]

print 'ROC AUC Score on the training examples %f ' %(roc_auc_score(y_train, predsTrain))
print 'ROC AUC Score on the test examples %f ' %(roc_auc_score(y_test, predsTest))