# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 13:50:57 2016

@author: abhishek
"""


import cPickle

# load engineered datasets
with open('../data/train_processed.pkl', 'r') as infile:
    train = cPickle.load(infile)
    infile.close()

with open('../data/test_processed.pkl', 'r') as infile:
    test = cPickle.load(infile)
    infile.close()
    

