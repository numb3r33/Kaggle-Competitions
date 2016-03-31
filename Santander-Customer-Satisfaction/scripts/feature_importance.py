# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 08:57:02 2016

@author: abhishek
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

X = train[train.columns.drop('TARGET')]
y = train.TARGET

forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
#plt.figure()
plt.title("Feature importances (RF)")
plt.bar(range(10), importances[indices][:10],
       color="r", yerr=std[indices][:10], align="center")
plt.xticks(range(10), train.columns[indices[:10]], rotation=90)
plt.xlim([-1, 10])
plt.show()