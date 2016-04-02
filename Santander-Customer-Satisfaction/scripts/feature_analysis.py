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
                self.train[col] = self.train[col].replace(missing_values[0], strategy(self.train[col]))
                
                self.test['is_missing_%s' %(col)] = (self.test[col] == missing_values[0]).astype(int)                
                self.test[col] = self.test[col].replace(missing_values[0], strategy(self.test[col]))
            
            elif (self.train[col] == missing_values[1]).any():
                self.train['is_missing_%s' %(col)] = (self.train[col] == missing_values[1]).astype(int)
                self.train[col] = self.train[col].replace(missing_values[1], strategy(self.train[col]))
                
                self.test['is_missing_%s' %(col)] = (self.test[col] == missing_values[1]).astype(int)
                self.test[col] = self.test[col].replace(missing_values[1], strategy(self.test[col]))
    
    def log_transformation(self):
        
        self.train = self.train[self.features].applymap(np.log1p)
        self.test = self.test[self.features].applymap(np.log1p)
    
    def discretize(self):
        self.train = self.train[self.features].astype(np.int)
        self.test = self.test[self.features].astype(np.int)
        
    def preprocess(self, impute_strategy):
        self.impute_missing_values(impute_strategy)
        self.log_transformation()
        self.discretize()
        
        
                
            
                