# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 22:33:15 2016

@author: abhishek
"""

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix

import xgboost as xgb

np.random.seed(44)

train = pd.read_csv('./data/train_processed_handle_na.csv')
test = pd.read_csv('./data/test_processed_handle_na.csv')

X = train[train.columns.drop('TARGET')]
y = train.TARGET

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1279)

# create model pipeline
scaler = MinMaxScaler()
select = SelectKBest(chi2, k=200)
clf = xgb.XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, min_child_weight=2, 
                        colsample_bytree=0.95, subsample=0.8, seed=1729)

pipeline = Pipeline([('scaler', scaler), ('select', select), ('clf', clf)])

pipeline.fit(X_train, y_train)

predsTrain = pipeline.predict_proba(X_train)[:, 1]
predsTest = pipeline.predict_proba(X_test)[:, 1]

print 'predictions on the training set %f ' %(roc_auc_score(y_train, predsTrain))
print 'predictions on the test set %f ' %(roc_auc_score(y_test, predsTest))

pipeline.fit(X, y)

predictions = pipeline.predict_proba(test)[:, 1]

submission = pd.read_csv('./data/sample_submission.csv')
submission.loc[:, 'TARGET'] = predictions
submission.to_csv('./submissions/booster_predictions_handle_na.csv', index=False)