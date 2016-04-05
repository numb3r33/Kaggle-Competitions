# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 08:38:24 2016

@author: abhishek
"""

from __future__ import division
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import pandas as pd
import numpy as np

# load train and test files

train = pd.read_csv('./data/train.csv', index_col='ID')
test = pd.read_csv('./data/test.csv', index_col='ID')


# set random seed
np.random.seed(10)

X = train[train.columns.drop('TARGET')]
y = train.TARGET

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.2, random_state=44)

n_folds = 10

skf = list(StratifiedKFold(ytrain, n_folds))

clfs = [XGBClassifier(n_estimators=147, learning_rate=0.1, min_child_weight=2, colsample_bytree=0.9, subsample=0.95, seed=1279),
        XGBClassifier(n_estimators=264, learning_rate=0.05, min_child_weight=2, colsample_bytree=0.9, subsample=0.9, seed=1729)]

print 'Creating train and test sets for blending'
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((test.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    print j, clf
    dataset_blend_test_j = np.zeros((test.shape[0], len(skf)))
    for i, (train, test_) in enumerate(skf):
        print "Fold", i
        X_train = X.values[train]
        y_train = y.values[train]
        X_test = X.values[test_]
        y_test = y.values[test_]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:,1]
        dataset_blend_train[test_, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(test)[:,1]
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

print
print "Blending."
clf = LogisticRegression()
clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:,1]

print "Linear stretch of predictions to [0,1]"
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())


#print 'ROC AUC Score on test set %f ' %(roc_auc_score(ytest, y_submission))

submission_df = pd.read_csv('./data/sample_submission.csv')
submission_df['TARGET'] = y_submission
submission_df.to_csv('./submissions/blend_two_xgboost.csv', index=False)