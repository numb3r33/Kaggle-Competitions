# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:24:09 2016

@author: abhishek
"""

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
import xgboost as xgb

np.random.seed(44)

train = pd.read_csv('./data/train_processed.csv')
test = pd.read_csv('./data/test_processed.csv')

X = train[train.columns.drop('TARGET')]
y = train.TARGET

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1279)

# evaluate xgboost model
param = dict([('max_depth', 3), ('learning_rate', 0.05), ('min_child_weight', 5), 
             ('colsample_bytree', 0.8), ('objective', 'binary:logistic'),
             ('eval_metric', 'auc'), ('subsample', 0.9), ('seed', 1729)])

dtrain = xgb.DMatrix(X_train.values, label=y_train.values)

num_round = 200

xgb.cv(param, dtrain, num_round, nfold=3, seed = 0)