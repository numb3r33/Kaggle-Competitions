from scipy import sparse
import numpy as np
from sklearn.base import BaseEstimator
import pandas as pd

class FeatureStacker(BaseEstimator):
	def __init__(self, transformer_list):
		self.transformer_list = transformer_list

	def get_feature_names(self):
		pass

	def fit(self, X, y=None):
		for name, trans in self.transformer_list:
			trans.fit(X, y)
		return self

	def transform(self, X):
		features = []
		for name, trans in self.transformer_list:
			features.append(trans.transform(X))

		issparse = [sparse.issparse(f) for f in features]

		if np.any(issparse):
			features = sparse.hstack(features).tocsr()
		else:
			features = np.hstack(features)

		return features

	def get_params(self, deep=True):
		if not deep:
			return super(FeatureStacker, self).get_params(deep=False)

		else:
			out = dict(self.transformer_list)

			for name, trans in self.transformer_list:
				for key, value in trans.get_params(deep=True).iteritems():
					out['%s__%s' % (name, key)] = value

			return out
