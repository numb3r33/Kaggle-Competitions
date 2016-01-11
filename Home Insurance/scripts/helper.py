from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import RandomizedSearchCV

from collections import defaultdict


import pandas as pd

def encode_labels(train, test):
	"""
	Encodes the categorical features into numerical features
	for both train and test dataframes
	"""

	categorical_features = train.select_dtypes(['object']).columns

	for col in categorical_features:
		total_values = pd.concat([train[col], test[col]], axis=0)
		
		lbl = LabelEncoder()
		
		lbl.fit(total_values)
		train[col] = lbl.transform(train[col])
		test[col] = lbl.transform(test[col])

	return train, test

def cv_optimize(X, y, cv, clf, parameters):
	"""
	Randomized Grid search on the parameter space to find out the best
	parameter settings to produce an accurate model
	"""

	rs = RandomizedSearchCV(clf, param_distributions=parameters, cv=cv, scoring='roc_auc')
	rs.fit(X, y)

	return rs

def transform_for_ranked(preds, index):
	ranks = []

	for i, pred in enumerate(preds):
		ranks.append((index[i], pred))

	return ranks


def ranked_averaging(predictions):
	all_ranks = defaultdict(list)

	for i, preds in enumerate(predictions):
		individual_ranks = []

		for e, pred in enumerate(preds):
			individual_ranks.append( (float(pred[1]), e, pred[0]) )

		for rank, item in enumerate( sorted(individual_ranks) ) :
			all_ranks[(item[1], item[2])].append(rank)

	average_ranks = []

	for k in sorted(all_ranks):
		average_ranks.append((sum(all_ranks[k])/len(all_ranks[k]),k))

	ranked_ranks = []

	for rank, k in enumerate(sorted(average_ranks)):
		ranked_ranks.append((k[1][0],k[1][1],(rank * 1.)/(len(average_ranks)-1)))
	
	return sorted(ranked_ranks)
