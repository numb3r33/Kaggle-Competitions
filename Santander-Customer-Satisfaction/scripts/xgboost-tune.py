# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 22:33:15 2016

@author: abhishek
"""

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
import xgboost as xgb

np.random.seed(44)

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

X = train[train.columns.drop('TARGET')]
y = train.TARGET

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1279)

# evaluate xgboost model
param = dict([('max_depth', 3), ('learning_rate', 0.1), ('colsample_bytree', 0.9),
              ('subsample', 0.95),
              ('objective', 'binary:logistic'),
             ('eval_metric', 'auc'), ('seed', 1729)])

dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
dtest = xgb.DMatrix(X_test.values, label=y_test.values)

watchlist = [(dtest, 'eval', (dtrain, 'train'))]

num_round = 100000

bst = xgb.train(param, dtrain, num_round, watchlist)