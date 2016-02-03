from sklearn.base import BaseEstimator
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import nltk

import pandas as pd
import numpy as np

class FeatureTransformer(BaseEstimator):
	"""
	Generate features
	"""

	def __init__(self):
		self.stopwords = stop_words.ENGLISH_STOP_WORDS
		self.stemmer = nltk.stem.PorterStemmer()

	def get_feature_names(self):
		feature_names = []

		feature_names.extend(self.truncated_svd.get_feature_names())
		return np.array(feature_names)

	def fit(self, X, y=None):
		self.fit_transform(X, y)

		return self

	def fit_transform(self, X, y=None):

		corpus = X.apply(lambda x: '%s %s' %(x['search_term'], x['product_title']), axis=1)
		words_lower = self._preprocess(corpus)
		
		empty_analyzer = lambda x: x
		self.unigram_vect = TfidfVectorizer(analyzer=empty_analyzer, min_df=3)

		unigrams = self.unigram_vect.fit_transform(words_lower)

		# self.truncated_svd = TruncatedSVD(n_components=50)
		# reduced_features = self.truncated_svd.fit_transform(unigrams)
		
		features = []
		features.append(unigrams.toarray())
		features = np.hstack(features)

		return features

	def _preprocess(self, corpus):
		corpus = list(corpus)
		words_lower = [' '.join([q.lower() for q in query.split(' ') if q not in self.stopwords]) for query in corpus ]
		return words_lower

	def transform(self, X):
		
		corpus = X.apply(lambda x: '%s %s' %(x['search_term'], x['product_title']), axis=1)
		words_lower = self._preprocess(corpus)
		unigrams = self.unigram_vect.transform(words_lower)
		# reduced_features = self.truncated_svd.transform(unigrams)

		features = []
		features.append(unigrams.toarray())
		features = np.hstack(features)
		
		return features
	
