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

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# Feature Analysis

# 4. imp_op_var39_comer_ult1 ( needs log transformation )
train['log_imp_op_var39_comer_ult1'] = train.imp_op_var39_comer_ult1.map(np.log1p)
sns.FacetGrid(train[train.imp_op_var39_comer_ult1>0.0], hue='TARGET', size=4)\
.map(sns.kdeplot, 'log_imp_op_var39_comer_ult1')\
.add_legend();

# 5. log_imp_op_var39_comer_ult3 ( needs log transformation )
train['log_imp_op_var39_comer_ult3'] = train.imp_op_var39_comer_ult3.map(np.log1p)
sns.FacetGrid(train[train.imp_op_var39_comer_ult3>0.0], hue='TARGET', size=4)\
.map(sns.kdeplot, 'log_imp_op_var39_comer_ult3')\
.add_legend();

# 6. imp_op_var40_comer_ult1 ( log transform )
train['log_train.imp_op_var40_comer_ult1'] = train.imp_op_var40_comer_ult1.map(np.log1p)
sns.FacetGrid(train[train.imp_op_var40_comer_ult1>0.0], hue='TARGET', size=4)\
.map(sns.kdeplot, 'log_imp_op_var40_comer_ult1')\
.add_legend();

# 7. imp_op_var40_comer_ult3
train['log_imp_op_var40_comer_ult3'] = train.imp_op_var40_comer_ult3.map(np.log1p)
sns.FacetGrid(train[train.imp_op_var40_comer_ult3>0.0], hue='TARGET', size=4)\
.map(sns.kdeplot, 'log_imp_op_var40_comer_ult3')\
.add_legend();

# 8. imp_op_var40_efect_ult1
train['log_imp_op_var40_efect_ult1'] = train.imp_op_var40_efect_ult1.map(np.log1p)

sns.FacetGrid(train[train.imp_op_var40_efect_ult1>0.0], hue='TARGET', size=4)\
.map(sns.kdeplot, 'log_imp_op_var40_efect_ult1')\
.add_legend();

# 9. 