from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import RandomizedSearchCV


import pandas as pd

def get_stratified_sample(df, target, percentage=.1):
	"""
	Returns a subset of the original dataframe based on
	stratified shuffle split which maintains the class balance
	in the subset
	"""

	sss = StratifiedShuffleSplit(target, train_size=percentage)
	train_index, test_index = next(iter(sss))

	return df.iloc[train_index]

def encode_labels(train, test):
	"""
	Encodes the categorical features into numerical features
	for both train and test dataframes
	"""

	categorical_features_train = train.select_dtypes(include=['object'])
	categorical_features_test = train.select_dtypes(include=['object'])

	categorical_features = categorical_features_train.columns

	categorical_features_train = categorical_features_train[categorical_features]
	categorical_features_test = categorical_features_test[categorical_features]

	for col in categorical_features:
		total_values = pd.concat([categorical_features_train[col], categorical_features_test[col]], axis=0)
		
		lbl = LabelEncoder()
		
		lbl.fit(total_values)
		categorical_features_train[col] = lbl.transform(categorical_features_train[col])
		categorical_features_test[col] = lbl.transform(categorical_features_test[col])

	train[categorical_features] = categorical_features_train
	test[categorical_features] = categorical_features_test

	return train, test

def cv_optimize(X, y, cv, clf, parameters):
	"""
	Randomized Grid search on the parameter space to find out the best
	parameter settings to produce an accurate model
	"""

	rs = RandomizedSearchCV(clf, param_distributions=parameters, cv=cv, scoring='roc_auc')
	rs.fit(X, y)

	return rs