# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 13:50:57 2016

@author: abhishek
"""


import pandas as pd
import numpy as np
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from search_map import spell_check_dict


stemmer = PorterStemmer()

# load train and test set
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# load product description and atttributes data
description = pd.read_csv('../data/product_descriptions.csv')
attributes = pd.read_csv('../data/attributes.csv')

def stem_words(sentence):
    return ' '.join([stemmer.stem(word) for word in sentence.split(' ')])
    

## NOTE:
## graders were shown images instead of product descriptions
## and other attributes

train = pd.merge(train, description, on='product_uid', how='left')
test = pd.merge(test, description, on='product_uid', how='left')

def correct_term(q):
    if q in spell_check_dict:
        return spell_check_dict[q]
    else:
        return q

# correct search queries
train.loc[:, 'search_term'] = train.search_term.map(correct_term)
test.loc[:, 'search_term'] = test.search_term.map(correct_term)


## remove non-alphanumeric characters
train.loc[:, 'product_description'] = train.product_description.map(lambda x: re.sub(r'[^A-Za-z0-9 ]', 
                                                                    ' ', x))

train.loc[:, 'search_term'] = train.search_term.map(lambda x: re.sub(r'[^A-Za-z0-9 ]', 
                                                                    ' ', x))

test.loc[:, 'product_description'] = test.product_description.map(lambda x: re.sub(r'[^A-Za-z0-9 ]', 
                                                                    ' ', x))

test.loc[:, 'search_term'] = test.search_term.map(lambda x: re.sub(r'[^A-Za-z0-9 ]', 
                                                                    ' ', x))

train.loc[:, 'product_description'] = train.product_description.map(stem_words)
train.loc[:, 'search_term'] = train.search_term.map(stem_words)

test.loc[:, 'product_description'] = test.product_description.map(stem_words)
test.loc[:, 'search_term'] = test.product_description.map(stem_words)


# corpus
corpus = train.apply(lambda x: '%s %s' %(x['product_title'].lower(), x['search_term'].lower()), axis=1)
corpus_test = test.apply(lambda x: '%s %s' %(x['product_title'].lower(), x['search_term'].lower()), axis=1)

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
corpus = tfidf.fit_transform(corpus.values)
corpus_test = tfidf.transform(corpus_test.values)

svd = TruncatedSVD(n_components=200)
corpus_svd = svd.fit_transform(corpus)
corpus_test_svd = svd.transform(corpus_test)

