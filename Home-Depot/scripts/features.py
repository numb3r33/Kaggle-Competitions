
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import LabelEncoder


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

		# how detailed the description is
		num_sentences_description = X['product_description'].map(self._num_sentences).reshape(-1, 1)

		X['product_description'] = X['product_description'].map(self._remove_stopwords)
		X['product_description'] = X['product_description'].map(self._add_periods)
		X['product_description'] = X['product_description'].map(self._stem_words)
		X['product_description'] = X['product_description'].map(self._preprocess)

		X['brand'] = X['brand'].map(self._remove_stopwords)
		X['brand'] = X['brand'].map(self._stem_words)
		X['brand'] = X['brand'].map(self._preprocess)

		# count frequency of the search term, product title and brand
		search_term_freq = nltk.FreqDist(X['search_term'])
		product_title_freq = nltk.FreqDist(X['product_title'])
		brand_freq = nltk.FreqDist(X['brand'])

		# corpus of text
		# corpus = X.apply(lambda x: '%s %s %s' %(x['search_term'], x['product_title'], x['brand']), axis=1)
		# self.unigram_vect = TfidfVectorizer(stop_words='english')
		# unigrams = self.unigram_vect.fit_transform(corpus)

		# self.truncated_svd = TruncatedSVD(n_components=25, random_state=1729)
		# reduced_features = self.truncated_svd.fit_transform(unigrams)



		is_query_in_title = X.apply(lambda x: self._contains_query_term(x['search_term'], x['product_title']), axis=1).reshape(-1, 1)
		is_query_in_description = X.apply(lambda x: self._contains_query_term(x['search_term'], x['product_description']), axis=1).reshape(-1, 1)
		is_query_in_brand = X.apply(lambda x: self._contains_query_term(x['search_term'], x['brand']), axis=1).reshape(-1, 1)
		
		jaccard_distance_search_title = X.apply(self._jaccard_distance_search_title, axis=1).reshape(-1, 1)
		jaccard_distance_search_description = X.apply(self._jaccard_distance_search_description, axis=1).reshape(-1, 1)
		jaccard_distance_search_brand = X.apply(self._jaccard_distance_search_brand, axis=1).reshape(-1, 1)
		
		query_length = X['search_term'].map(lambda x: len(x.split(' '))).reshape(-1, 1)
		title_length = X['product_title'].map(lambda x: len(x.split(' '))).reshape(-1, 1)
		brand_length = X['brand'].map(lambda x: len(x.split(' '))).reshape(-1, 1)
		description_length = X['product_description'].map(lambda x: len(x.split(' '))).reshape(-1, 1)
			
		# check to see if search term has dimensions

		# has_dimensions = X['search_term'].map(self._check_for_dimensions).reshape(-1, 1)

		# label search term, title and brand based on the frequency
		search_term_popularity = X['search_term'].map(lambda x: search_term_freq[x]).reshape(-1, 1)
		product_title_popularity = X['product_title'].map(lambda x: product_title_freq[x]).reshape(-1, 1)
		product_brand = X['brand'].map(lambda x: brand_freq[x]).reshape(-1, 1)

		features = []

		# features.append(reduced_features)
		features.append(is_query_in_title)
		features.append(is_query_in_description)
		features.append(is_query_in_brand)
		features.append(query_length)
		features.append(title_length)
		features.append(brand_length)
		features.append(description_length)
		features.append(jaccard_distance_search_title)
		features.append(jaccard_distance_search_description)
		features.append(jaccard_distance_search_brand)
		features.append(num_sentences_description)
		# features.append(has_dimensions)
		features.append(search_term_popularity)
		features.append(product_title_popularity)
		features.append(description_length)
		

		features = np.hstack(features)

		return features

	def _stem_words(self, sentence):
		return ' '.join([self.stemmer.stem(words) for words in sentence.split()])
	
	def _remove_stopwords(self, sentence):
		return ' '.join([re.sub(r'[^\w\s\d]','',word.lower()) for word in sentence.split() if word not in self.stopwords])

	def _preprocess(self, s):
		
		s = s.replace("'","in.") 
		s = s.replace("inches","in.") 
		s = s.replace("inch","in.")
		s = s.replace(" in ","in. ") 

		s = s.replace("''","ft.") 
		s = s.replace(" feet ","ft. ") 
		s = s.replace("feet","ft.") 
		s = s.replace("foot","ft.") 
		s = s.replace(" ft ","ft. ") 
		s = s.replace(" ft.","ft.") 

		s = s.replace(" pounds ","lb. ")
		s = s.replace(" pound ","lb. ") 
		s = s.replace("pound","lb.") 
		s = s.replace(" lb ","lb. ") 
		s = s.replace(" lb.","lb.") 
		s = s.replace(" lbs ","lb. ") 
		s = s.replace("lbs.","lb.") 

		s = s.replace(" x "," xby ")
		s = s.replace("*"," xby ")
		s = s.replace(" by "," xby")
		s = s.replace("x0"," xby 0")
		s = s.replace("x1"," xby 1")
		s = s.replace("x2"," xby 2")
		s = s.replace("x3"," xby 3")
		s = s.replace("x4"," xby 4")
		s = s.replace("x5"," xby 5")
		s = s.replace("x6"," xby 6")
		s = s.replace("x7"," xby 7")
		s = s.replace("x8"," xby 8")
		s = s.replace("x9"," xby 9")
		s = s.replace("0x","0 xby ")
		s = s.replace("1x","1 xby ")
		s = s.replace("2x","2 xby ")
		s = s.replace("3x","3 xby ")
		s = s.replace("4x","4 xby ")
		s = s.replace("5x","5 xby ")
		s = s.replace("6x","6 xby ")
		s = s.replace("7x","7 xby ")
		s = s.replace("8x","8 xby ")
		s = s.replace("9x","9 xby ")

		s = s.replace("1", "one")
		s = s.replace("2", "two")
		s = s.replace("3", "three")
		s = s.replace("4", "four")
		s = s.replace("5", "five")
		s = s.replace("6", "six")
		s = s.replace("7", "seven")
		s = s.replace("8", "eight")
		s = s.replace("9", "nine")
		s = s.replace("0", "zero")

		s = s.replace("/"," or ")		

		s = s.replace(" sq ft","sq.ft. ") 
		s = s.replace("sq ft","sq.ft. ")
		s = s.replace("sqft","sq.ft. ")
		s = s.replace(" sqft ","sq.ft. ") 
		s = s.replace("sq. ft","sq.ft. ") 
		s = s.replace("sq ft.","sq.ft. ") 
		s = s.replace("sq feet","sq.ft. ") 
		s = s.replace("square feet","sq.ft. ") 

		s = s.replace(" gallons ","gal. ") 
		s = s.replace(" gallon ","gal. ") 
		s = s.replace("gallons","gal.") 
		s = s.replace("gallon","gal.") 
		s = s.replace(" gal ","gal. ") 
		s = s.replace(" gal","gal.") 

		s = s.replace("ounces","oz.")
		s = s.replace("ounce","oz.")
		s = s.replace(" oz.","oz. ")
		s = s.replace(" oz ","oz. ")

		s = s.replace("centimeters","cm.")    
		s = s.replace(" cm.","cm.")
		s = s.replace(" cm ","cm. ")

		s = s.replace("milimeters","mm.")
		s = s.replace(" mm.","mm.")
		s = s.replace(" mm ","mm. ")

		s = s.replace("degrees","deg. ")
		s = s.replace("degree","deg. ")

		s = s.replace("volts","volt. ")
		s = s.replace("volt","volt. ")

		s = s.replace("watts","watt. ")
		s = s.replace("watt","watt. ")

		s = s.replace("amps","amp. ")
		s = s.replace(" amp ","amp. ")

		s = s.replace("&nbsp", " ")

		s = s.replace("toliet","toilet")
		s = s.replace("airconditioner","air conditioner")
		s = s.replace("vinal","vinyl")
		s = s.replace("vynal","vinyl")
		s = s.replace("snowbl","snow bl")
		s = s.replace("plexigla","plexi gla")
		s = s.replace("rustoleum","rust-oleum")
		s = s.replace("whirpool","whirlpool")
		s = s.replace("whirlpoolga", "whirlpool ga")

		return s

	def _jaccard_distance_search_title(self, df):
		search_term = set(df['search_term'])
		product_title = set(df['product_title'])

		return len(search_term & product_title) * 1. / (len(search_term | product_title) + 1)
	
	def _jaccard_distance_search_description(self, df):
		search_term = set(df['search_term'])
		product_description = set(df['product_description'])

		return len(search_term & product_description) * 1. / (len(search_term | product_description) + 1)

	def _jaccard_distance_search_brand(self, df):
		search_term = set(df['search_term'])
		product_brand = set(df['brand'])

		return len(search_term & product_brand) * 1. / (len(search_term | product_brand) + 1)

	def _check_for_dimensions(self, search_term):
		dimensions_regex = r'[0-9]+[ ]*x[ ]*[0-9]+[ ]*(?:x[ ]*[0-9]+)?'

		return int(len(re.findall(dimensions_regex, search_term)) > 0)

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

		num_sentences_description = X['product_description'].map(self._num_sentences).reshape(-1, 1)

		X['product_description'] = X['product_description'].map(self._remove_stopwords)
		X['product_description'] = X['product_description'].map(self._add_periods)
		X['product_description'] = X['product_description'].map(self._stem_words)
		X['product_description'] = X['product_description'].map(self._preprocess)

		X['brand'] = X['brand'].map(self._remove_stopwords)
		X['brand'] = X['brand'].map(self._stem_words)
		X['brand'] = X['brand'].map(self._preprocess)

		# count frequency of the search term, product title and brand
		search_term_freq = nltk.FreqDist(X['search_term'])
		product_title_freq = nltk.FreqDist(X['product_title'])
		brand_freq = nltk.FreqDist(X['brand'])

		# corpus of text
		# corpus = X.apply(lambda x: '%s %s %s' %(x['search_term'], x['product_title'], x['brand']), axis=1)
		# unigrams = self.unigram_vect.transform(corpus)
		# reduced_features = self.truncated_svd.transform(unigrams)



		is_query_in_title = X.apply(lambda x: self._contains_query_term(x['search_term'], x['product_title']), axis=1).reshape(-1, 1)
		is_query_in_description = X.apply(lambda x: self._contains_query_term(x['search_term'], x['product_description']), axis=1).reshape(-1, 1)
		is_query_in_brand = X.apply(lambda x: self._contains_query_term(x['search_term'], x['brand']), axis=1).reshape(-1, 1)
		jaccard_distance_search_title = X.apply(self._jaccard_distance_search_title, axis=1).reshape(-1, 1)
		jaccard_distance_search_description = X.apply(self._jaccard_distance_search_description, axis=1).reshape(-1, 1)
		jaccard_distance_search_brand = X.apply(self._jaccard_distance_search_brand, axis=1).reshape(-1, 1)
		query_length = X['search_term'].map(lambda x: len(x.split(' '))).reshape(-1, 1)
		title_length = X['product_title'].map(lambda x: len(x.split(' '))).reshape(-1, 1)
		brand_length = X['brand'].map(lambda x: len(x.split(' '))).reshape(-1, 1)
		description_length = X['product_description'].map(lambda x: len(x.split(' '))).reshape(-1, 1)
		
		# check to see if search term has dimensions

		# has_dimensions = X['search_term'].map(self._check_for_dimensions).reshape(-1, 1)

		# label search term, title and brand based on the frequency
		search_term_popularity = X['search_term'].map(lambda x: search_term_freq[x]).reshape(-1, 1)
		product_title_popularity = X['product_title'].map(lambda x: product_title_freq[x]).reshape(-1, 1)
		product_brand = X['brand'].map(lambda x: brand_freq[x]).reshape(-1, 1)

		features = []

		# features.append(reduced_features)
		features.append(is_query_in_title)
		features.append(is_query_in_description)
		features.append(is_query_in_brand)
		features.append(query_length)
		features.append(title_length)
		features.append(brand_length)
		features.append(description_length)
		features.append(jaccard_distance_search_title)
		features.append(jaccard_distance_search_description)
		features.append(jaccard_distance_search_brand)
		features.append(num_sentences_description)
		# features.append(has_dimensions)
		features.append(search_term_popularity)
		features.append(product_title_popularity)
		features.append(description_length)
		

		features = np.hstack(features)

		return features
