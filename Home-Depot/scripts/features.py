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
		self.unigram_vect = TfidfVectorizer(analyzer=empty_analyzer, ngram_range=(1, 2), min_df=3)

		unigrams = self.unigram_vect.fit_transform(words_lower)
		is_query_in_title = self._contains_query_term(X['search_term'], X['product_title'])

		# self.truncated_svd = TruncatedSVD(n_components=100)
		# reduced_features = self.truncated_svd.fit_transform(unigrams)
		
		features = []
		# features.append(reduced_features)
		features.append(unigrams.toarray())
		features.append(is_query_in_title)
		features = np.hstack(features)

		return features

	def _preprocess(self, corpus):
		corpus = list(corpus)
		words_lower = [' '.join([q.lower() for q in query.split(' ') if q not in self.stopwords]) for query in corpus ]
		return words_lower

	def _contains_query_term(self, search_terms, product_title):
		is_query_in_title = []
		
		for i in range(len(search_terms)):
			titles = product_title.irow(i)
			query_terms = search_terms.irow(i).split(' ')

			contains_terms = False

			for term in query_terms:
				if term in titles.lower():
					contains_terms = True
					
			if contains_terms:
				is_query_in_title.append(1)
			else:
				is_query_in_title.append(0)

		return np.array([is_query_in_title]).T
		
	def transform(self, X):
		
		corpus = X.apply(lambda x: '%s %s' %(x['search_term'], x['product_title']), axis=1)
		words_lower = self._preprocess(corpus)
		unigrams = self.unigram_vect.transform(words_lower)
		is_query_in_title = self._contains_query_term(X['search_term'], X['product_title'])

		# reduced_features = self.truncated_svd.transform(unigrams)

		features = []
		# features.append(reduced_features)
		features.append(unigrams.toarray())
		features.append(is_query_in_title)
		features = np.hstack(features)
		
		return features
	
