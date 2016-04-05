# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:27:37 2016

@author: abhishek
"""

import numpy as np
import pandas as pd

from scipy.cluster import vq


# load train and test set
train = pd.read_csv('./data/train.csv', index_col='ID')
test = pd.read_csv('./data/test.csv', index_col='ID')

# columns with high frequency values
high_frequency = [col for col in train.columns if len(train[col].unique()) > 10]

for col in high_frequency:
    codebook = vq.kmeans(train[col].values.astype(float), 5)
    train_values = []
    test_values = []

    for val in train[col]:
        train_values.append(vq.vq(val, codebook[0])[0][0])
    
    for val in test[col]:
        test_values.append(vq.vq(val, codebook[0])[0][0])
    
    train[col] = np.array(train_values)
    test[col] = np.array(test_values)
    
train.to_csv('./data/synthesized/train_vq.csv', index=False)
test.to_csv('./data/synthesized/test_vq.csv', index=False)