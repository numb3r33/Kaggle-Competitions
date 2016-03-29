# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 07:52:35 2016

@author: abhishek
"""

import pandas as pd

## Evaluation metric is AUC

# load train and test files
train = pd.read_csv('data/train.csv', index_col='ID')
test = pd.read_csv('data/test.csv', index_col='ID')


## NOTES
##
## 1. 9999999999 to mark missing values
## 2. -999999 to mark missing values

## need to remove some features because they are either constant or
## identical to other column

def get_constant_features(df):
    columns = df.columns
    
    constant_features = []
    
    for col in columns:
        if df[col].std() == 0.0:
            constant_features.append(col)
    
    return constant_features

def get_identical_features(df):
    columns  = df.columns
    
    identical_features = []
    
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if (df[columns[i]] == df[columns[j]]).all():
                identical_features.append(columns[i])
    
    identical_features = set(identical_features)
    identical_features = list(identical_features)
    
    return identical_features

def concat_features(constant_features, identical_features):
    features_to_remove = []
    
    for col in constant_features:
        features_to_remove.append(col)
    
    for col in identical_features:
        features_to_remove.append(col)
        
    return features_to_remove

constant_features = get_constant_features(train)
identical_features = get_identical_features(train)
features_to_remove = concat_features(constant_features, identical_features)

## var 3 has missing value ( -999999 )
## 26 more features with missing values
## Here is the list

# Index([u'delta_imp_amort_var18_1y3', u'delta_imp_amort_var34_1y3',
#       u'delta_imp_aport_var13_1y3', u'delta_imp_aport_var17_1y3',
#       u'delta_imp_aport_var33_1y3', u'delta_imp_compra_var44_1y3',
#       u'delta_imp_reemb_var13_1y3', u'delta_imp_reemb_var17_1y3',
#       u'delta_imp_reemb_var33_1y3', u'delta_imp_trasp_var17_in_1y3',
#       u'delta_imp_trasp_var17_out_1y3', u'delta_imp_trasp_var33_in_1y3',
#       u'delta_imp_trasp_var33_out_1y3', u'delta_imp_venta_var44_1y3',
#       u'delta_num_aport_var13_1y3', u'delta_num_aport_var17_1y3',
#       u'delta_num_aport_var33_1y3', u'delta_num_compra_var44_1y3',
#       u'delta_num_reemb_var13_1y3', u'delta_num_reemb_var17_1y3',
#       u'delta_num_reemb_var33_1y3', u'delta_num_trasp_var17_in_1y3',
#       u'delta_num_trasp_var17_out_1y3', u'delta_num_trasp_var33_in_1y3',
#       u'delta_num_trasp_var33_out_1y3', u'delta_num_venta_var44_1y3'],
#      dtype='object')
some_more_features_with_constant_value = ['delta_num_trasp_var33_out_1y3', 'delta_num_reemb_var33_1y3',
                                          'delta_imp_trasp_var33_out_1y3', 'delta_imp_reemb_var33_1y3',
                                          'delta_imp_amort_var34_1y3', 'delta_imp_amort_var18_1y3']

for feat_name in some_more_features_with_constant_value:
    features_to_remove.append(feat_name)

# treat var3 differently
train.loc[:, 'missing_value_var3'] = (train.var3 == -999999).astype(int)
train.loc[:, 'var3'] = train.var3.fillna(train.var3.mode())

test.loc[:, 'missing_value_var3'] = (test.var3 == -999999).astype(int)
test.loc[:, 'var3'] = test.var3.fillna(train.var3.mode())

# remove features
features = train.columns.drop(features_to_remove)

train_subset = train[features]

features = features.drop('TARGET')
test_subset = test[features]

train_subset.to_csv('./data/train_processed.csv', index=False)
test_subset.to_csv('./data/test_processed.csv', index=False)
