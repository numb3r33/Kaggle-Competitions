from sklearn.base import BaseEstimator
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression


import nltk

import pandas as pd
import numpy as np

import re

class FeatureTransformer(BaseEstimator):
	"""
	Generate features
	"""

	def __init__(self):
		self.stopwords = stop_words.ENGLISH_STOP_WORDS
		self.stemmer = nltk.stem.SnowballStemmer('english')

	def get_feature_names(self):
		feature_names = []

		feature_names.extend(self.truncated_svd.get_feature_names())
		return np.array(feature_names)

	def fit(self, X, y=None):
		self.fit_transform(X, y)

		return self

	def fit_transform(self, X, y=None):
		
		X['search_term'] = X['search_term'].map(self._remove_stopwords)
		X['search_term'] = X['search_term'].map(self._stem_words)
		X['search_term'] = X['search_term'].map(self._preprocess)

		X['product_title'] = X['product_title'].map(self._remove_stopwords)
		X['product_title'] = X['product_title'].map(self._stem_words)
		X['product_title'] = X['product_title'].map(self._preprocess)

		num_sentences = X['product_description'].map(self._num_sentences).reshape(-1, 1)

		X['product_description'] = X['product_description'].map(self._remove_stopwords)
		X['product_description'] = X['product_description'].map(self._add_periods)
		X['product_description'] = X['product_description'].map(self._stem_words)
		X['product_description'] = X['product_description'].map(self._preprocess)


		# corpus = X.apply(lambda x: '%s %s %s' %(x['search_term'], x['product_title'], x['product_description']), axis=1)
		# self.count_vect = CountVectorizer(analyzer='word', min_df=2)
		# bow = self.count_vect.fit_transform(corpus)

		
		is_query_in_title = X.apply(lambda x: self._contains_query_term(x['search_term'], x['product_title']), axis=1).reshape(-1, 1)
		is_query_in_description = X.apply(lambda x: self._contains_query_term(x['search_term'], x['product_description']), axis=1).reshape(-1, 1)
		query_length = self._get_query_length(X['search_term'])
		
		# self.selector = SelectKBest(f_regression, k=10)
		# reduced_features = self.selector.fit_transform(bow.todense(), X['relevance'])

		
		features = []
		# features.append(reduced_features)
		features.append(is_query_in_title)
		features.append(is_query_in_description)
		features.append(query_length)
		features.append(num_sentences)

		features = np.hstack(features)

		return features

	def _stem_words(self, sentence):
		return ' '.join([self.stemmer.stem(words) for words in sentence.split()])
	
	def _remove_stopwords(self, sentence):
		return ' '.join([re.sub(r'[^\w\s\d]','',word.lower()) for word in sentence.split() if word not in self.stopwords])

	def _preprocess(self, sentence):
		sentence = sentence.replace('x', ' times ')
		sentence = sentence.replace("'", ' inches ')
		sentence = sentence.replace('in.', ' inches ')
		sentence = sentence.replace('ft', ' feet ')
		sentence = sentence.replace('mm', ' milimeters ')
		sentence = sentence.replace('btu', 'british thermal unit ')
		sentence = sentence.replace('cc', ' cubic centimeters ')
		sentence = sentence.replace('cfm', ' cubic feet per minute ')
		sentence = sentence.replace('ga', ' gallons ')
		sentence = sentence.replace('lbs', ' pounds ')
		sentence = sentence.replace('*', ' times ')

		return sentence



	def _get_query_length(self, search_terms):
		query_length = search_terms.map(lambda x: len(x.split(' ')))

		return np.array([query_length]).T

	def _contains_query_term(self, needle, haystack):
		return sum(int(haystack.find(word)>=0) for word in needle.split())

	def _num_sentences(self, text):
		return len(text.split('.'))

	def _add_periods(self, sentence):
		def transform(matchobj):
			matched_group = matchobj.group(0)
			uppercase_index = re.search(r'[A-Z]', matched_group).start()
			return matched_group[:uppercase_index] + '. ' + matched_group[uppercase_index:]
		
		return re.sub(r'[a-z]+[A-Z][a-z]+', transform, sentence)

	
		
	def transform(self, X):
		X['search_term'] = X['search_term'].map(self._remove_stopwords)
		X['search_term'] = X['search_term'].map(self._stem_words)
		X['search_term'] = X['search_term'].map(self._preprocess)


		X['product_title'] = X['product_title'].map(self._remove_stopwords)
		X['product_title'] = X['product_title'].map(self._stem_words)
		X['product_title'] = X['product_title'].map(self._preprocess)
		
		num_sentences = X['product_description'].map(self._num_sentences).reshape(-1, 1)

		X['product_description'] = X['product_description'].map(self._remove_stopwords)
		X['product_description'] = X['product_description'].map(self._add_periods)
		X['product_description'] = X['product_description'].map(self._stem_words)
		X['product_description'] = X['product_description'].map(self._preprocess)


		# corpus = X.apply(lambda x: '%s %s %s' %(x['search_term'], x['product_title'], x['product_description']), axis=1)
		# bow = self.count_vect.transform(corpus)
		

		is_query_in_title = X.apply(lambda x: self._contains_query_term(x['search_term'], x['product_title']), axis=1).reshape(-1, 1)
		is_query_in_description = X.apply(lambda x: self._contains_query_term(x['search_term'], x['product_description']), axis=1).reshape(-1, 1)
		query_length = self._get_query_length(X['search_term'])
		

		# reduced_features = self.selector.transform(bow.todense())

		features = []
		# features.append(reduced_features)
		features.append(is_query_in_title)
		features.append(is_query_in_description)
		features.append(query_length)
		features.append(num_sentences)

		features = np.hstack(features)
		
		return features
	
