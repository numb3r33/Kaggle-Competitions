# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 14:30:44 2016

@author: abhishek
"""
import pandas as pd
import numpy as np
import re
from search_map import spell_check_dict
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse as sps

class Dataset(object):
    def __init__(self, train, test):
        self.train = train.copy()
        self.test = test.copy()
        
        self.y = train.relevance
        
        self.tfidf_vectorizer = TfidfVectorizer()
     
    def correct_search_terms(self, train, test):
        def correct_term(q):
            if q in spell_check_dict:
                return spell_check_dict[q]
            else:
                return q
        
        train_search_terms = train.search_term
        test_search_terms = test.search_term
  
        return train_search_terms, test_search_terms
    
    def stem_word(self, word):
        suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']
        
        for suffix in suffixes:
            if word.endswith(suffix):
                return word[:-len(suffix)]
        
        return word
    
    def tokenize(self, sentence):
        return word_tokenize(sentence)
    
    def stemming(self, sentence):
        tokens = self.tokenize(sentence)
        stemmed = ' '.join([self.stem_word(token) for token in tokens])
        
        return stemmed
    
    def filter_characters(self, char):
        return char == '\n' or 32 <= ord(char) <= 126

    def sanitize_title(self, sentence):
        return filter(self.filter_characters, sentence)
    
    def preprocessing(self, to_stem=False):
        corrected_q_train, corrected_q_test = self.correct_search_terms(self.train, self.test)
        
        self.train['search_term'] = corrected_q_train
        self.test['search_term'] = corrected_q_test
        
        self.train['search_term'] = self.train.search_term.map(lambda x: x.lower())
        self.test['search_term'] = self.test.search_term.map(lambda x: x.lower())
        
        
        self.train['product_title'] = self.train.product_title.map(self.sanitize_title)
        self.test['product_title'] = self.test.product_title.map(self.sanitize_title)
        
        if to_stem:
            self.train['search_term'] = self.train.search_term.map(self.stemming)
            self.test['search_term'] = self.test.search_term.map(self.stemming)
    
    def num_tokens_query(self, query):
        return len(word_tokenize(query))
    
    def num_tokens_title(self, title):
        return len(word_tokenize(title))
    
    def cosine_similarity_score(self, row):
        query = row['search_term']
        title = row['product_title']
        
        corpus = np.array([query, title])
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        
        normal_array = tfidf_matrix.toarray()
        
        query_repr = normal_array[0].reshape(-1, 1)
        title_repr = normal_array[1].reshape(-1, 1)
        
        return cosine_similarity(query_repr, title_repr)[0][0]
    
    def jaccard_score(self, row):
        query = row['search_term']
        title = row['product_title']
        
        corpus = np.array([query, title])
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        
        return jaccard_similarity_score(tfidf_matrix[0], tfidf_matrix[1])
        
    
    def numerical_features(self):
        """
        1. Number of tokens in the query
        2. Number of tokens in the title
        3. Cosine similarity between title and query
        """
        
        self.train['num_query_tokens'] = self.train.search_term.map(self.num_tokens_query)
        self.test['num_query_tokens'] = self.test.search_term.map(self.num_tokens_query)
        
        self.train['num_title_tokens'] = self.train.product_title.map(self.num_tokens_title)
        self.test['num_title_tokens'] = self.test.product_title.map(self.num_tokens_title)
        
        self.train['cosine_score'] = self.train.apply(self.cosine_similarity_score, axis=1)
        self.test['cosine_score'] = self.test.apply(self.cosine_similarity_score, axis=1)
    
    def text_features(self):
        corpus_train = self.train.apply(lambda x: '%s %s' %(x['product_title'], x['search_term']), axis=1)
        corpus_test = self.test.apply(lambda x: '%s %s' %(x['product_title'], x['search_term']), axis=1)

        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
        corpus = tfidf.fit_transform(corpus_train.values)
        corpus_test = tfidf.transform(corpus_test.values)

        svd = TruncatedSVD(n_components=200)
        
        self.corpus_svd = svd.fit_transform(corpus)
        self.corpus_test_svd = svd.transform(corpus_test)
    
    def combine_features(self):
        features = ['num_query_tokens', 'num_title_tokens', 'cosine_score']
        
        numerical_features = self.train[features]
        numerical_features_test = self.test[features]
        
        self.processed_features_train = sps.hstack([numerical_features, self.corpus_svd])
        self.processesd_features_test = sps.hstack([numerical_features_test, self.corpus_test_svd])

        
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

dataset = Dataset(train, test)
dataset.preprocessing()
dataset.text_features()
dataset.numerical_features()
dataset.combine_features()

