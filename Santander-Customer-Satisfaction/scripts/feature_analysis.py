# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 22:30:05 2016

@author: abhishek
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load train and test set

train = pd.read_csv('./data/train.csv', index_col='ID')
test = pd.read_csv('./data/test.csv', index_col='ID')

## Class that would represent different synthesized datasets

class Dataset():
    def __init__(self, train, test):
        self.train = train.copy()
        self.test = test.copy()
        self.features = train.columns[:-1]
    
    def impute_missing_values(self, strategy):
        missing_values = [-999999.0, 9999999999.0]
        
        for col in self.features:
            if (self.train[col] == missing_values[0]).any():  
                self.train['is_missing_%s' %(col)] = (self.train[col] == missing_values[0]).astype(int)                
                
                if strategy == 'mean':    
                    strategy_applied_value = self.train[self.train[col] != missing_values[0]][col].mean()             
                elif strategy == 'median':
                    strategy_applied_value = self.train[self.train[col] != missing_values[0]][col].median()
                else:
                    strategy_applied_value = self.train[self.train[col] != missing_values[0]][col].mode()
                    
                self.train[col] = self.train[col].replace(missing_values[0], strategy_applied_value)
                
                
                self.test['is_missing_%s' %(col)] = (self.test[col] == missing_values[0]).astype(int)                
                
                if strategy == 'mean':    
                    strategy_applied_value = self.test[self.test[col] != missing_values[0]][col].mean()               
                elif strategy == 'median':
                    strategy_applied_value = self.test[self.test[col] != missing_values[0]][col].median()
                else:
                    strategy_applied_value = self.test[self.test[col] != missing_values[0]][col].mode()
                                
                self.test[col] = self.test[col].replace(missing_values[0], strategy_applied_value)
            
            elif (self.train[col] == missing_values[1]).any():
                self.train['is_missing_%s' %(col)] = (self.train[col] == missing_values[1]).astype(int)
                
                if strategy == 'mean':    
                    strategy_applied_value = self.train[self.train[col] != missing_values[1]][col].mean()             
                elif strategy == 'median':
                    strategy_applied_value = self.train[self.train[col] != missing_values[1]][col].median()
                else:
                    strategy_applied_value = self.train[self.train[col] != missing_values[1]][col].mode()
                                
                self.train[col] = self.train[col].replace(missing_values[1], strategy_applied_value)

                
                self.test['is_missing_%s' %(col)] = (self.test[col] == missing_values[1]).astype(int)
                
                if strategy == 'mean':    
                    strategy_applied_value = self.test[self.test[col] != missing_values[1]][col]              
                elif strategy == 'median':
                    strategy_applied_value = self.test[self.test[col] != missing_values[1]][col]
                else:
                    strategy_applied_value = self.test[self.test[col] != missing_values[1]][col]
                
                self.test[col] = self.test[col].replace(missing_values[1], strategy_applied_value)
    
    def get_positive_valued_features(self):
        feature_status = (self.train < 0).any()
        neg_valued_features = feature_status[feature_status == True].index
        
        return self.features.drop(neg_valued_features)
        
    def log_transformation(self):
        self.non_neg_features = self.get_positive_valued_features()
        
        self.train[self.non_neg_features] = self.train[self.non_neg_features].applymap(np.log1p)
        self.test[self.non_neg_features] = self.test[self.non_neg_features].applymap(np.log1p)
    
    def discretize(self):
        self.train = self.train.astype(np.int)
        self.test = self.test.astype(np.int)
        
    def preprocess(self, impute_strategy):
        self.impute_missing_values(impute_strategy)
        self.log_transformation()
#        self.discretize()
        

dataset_mean = Dataset(train, test)
dataset_mean.preprocess('mean')

dataset_median = Dataset(train, test)
dataset_median.preprocess('median')

dataset_mode = Dataset(train, test)
dataset_mode.preprocess('mode')
                
            
                