from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

import pandas as pd
import numpy as np

class FeatureTransformer(BaseEstimator):
	"""
	Generate features
	"""

	def __init__(self):
		self.X = pd.read_csv('./data/train.csv', parse_dates=['Original_Quote_Date'], index_col='QuoteNumber')
		self.X_test = pd.read_csv('./data/test.csv', parse_dates=['Original_Quote_Date'], index_col='QuoteNumber')

		self.X = self.X.fillna(-1)
		self.X_test = self.X_test.fillna(-1)
		

	def get_feature_names(self):
		feature_names = []

		feature_names.extend(['year_original_quote', 'month_original_quote', 'weekday_original_quote'])
		feature_names.extend(self.categorical_features_columns)
		feature_names.extend(self.numerical_features_columns)

		return np.array(feature_names)

	def fit(self, X, y=None):
		self.fit_transform(X, y)

		return self

	def fit_transform(self, X, y=None):
		
		date_features = self._process_dates(X)
		is_nan_features = self._is_nan(X)
		count_nan_features = self._count_nans(X)
		categorical_features = self._process_categorical_features(X)
		numerical_features = self._process_numerical_features(X)

		features = []
		
		features.append(date_features)
		features.append(is_nan_features)
		features.append(count_nan_features)
		features.append(categorical_features)
		features.append(numerical_features)

		features = np.hstack(features)
		
		return features

	def _process_dates(self, X):
		'Extract year, month and weekday of original quote'

		year_original_quote = X.Original_Quote_Date.dt.year
		month_original_quote = X.Original_Quote_Date.dt.month
		weekday_original_quote = X.Original_Quote_Date.dt.weekday

		return np.array([year_original_quote, month_original_quote, weekday_original_quote]).T

	def _is_nan(self, X):
		'Check to see whether record has any nan value or not'
		null_check = X.apply(lambda x: -1 in x.values, axis=1) * 1.

		return np.array(null_check).reshape(-1, 1)

	def _count_nans(self, X):
		'Count number of missing values in a quote'

		count_nans = X.apply(lambda x: list(x.values).count(-1), axis=1)

		return np.array([count_nans]).T

	def _process_categorical_features(self, X):
		'Encode categorical features into numerical features'

		self.categorical_features_columns = X.select_dtypes(['object']).columns
		categorical_features = []

		for cat in self.categorical_features_columns:
			lbl = LabelEncoder()

			lbl.fit(pd.concat([self.X[cat], self.X_test[cat]], axis=0))

			categorical_features.append(lbl.transform(X[cat]))

		return np.array(categorical_features).T

	def _process_numerical_features(self, X):
		'Return numerical features as it is'

		self.numerical_features_columns = X.select_dtypes(['int32', 'int64', 'float32', 'float64'])

		numerical_features = []

		for col in self.numerical_features_columns:
			numerical_features.append(X[col])

		return np.array(numerical_features).T
		

	def transform(self, X):
		date_features = self._process_dates(X)
		is_nan_features = self._is_nan(X)
		count_nan_features = self._count_nans(X)
		categorical_features = self._process_categorical_features(X)
		numerical_features = self._process_numerical_features(X)

		features = []
		
		features.append(date_features)
		features.append(is_nan_features)
		features.append(count_nan_features)
		features.append(categorical_features)
		features.append(numerical_features)

		features = np.hstack(features)
		
		return features
	
