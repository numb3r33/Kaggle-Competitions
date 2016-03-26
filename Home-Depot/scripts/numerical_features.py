# -*- coding: utf-8 -*-
"""
Kaggle Home Depot Challenge

Evaluation Metric : RMSE
"""
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction import text
from difflib import SequenceMatcher as seq_matcher

pattern = re.compile(r'\b(' + r'|'.join(text.ENGLISH_STOP_WORDS) + r')\b\s*')


# load train and test set
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# most of the queries are relevant
search_term_frequency_train = train.groupby('product_uid').size().reset_index()
search_term_frequency_train.columns = ['product_uid', 'query_frequency']

search_term_frequency_test = test.groupby('product_uid').size().reset_index()
search_term_frequency_test.columns = ['product_uid', 'query_frequency']

# merge this with train and test set
train = pd.merge(train, search_term_frequency_train, on='product_uid', how='left')
test = pd.merge(test, search_term_frequency_test, on='product_uid', how='left')

# some preprocessing functions
def filter_characters(char):
    return char == '\n' or 32 <= ord(char) <= 126

def sanitize(s):
    s = s.replace('ft.', 'feet')
    s = s.replace('cu.', 'cubic')
    s = s.replace('mm', 'milimeters')
    s = s.replace('oz.', 'ounces')
    s = s.replace('btu', 'british thermal unit')
    s = s.replace('otr', 'over the range')
    s = s.replace('lb.', 'pounds')
    s = s.replace('in.', 'inches')
    s = s.replace('&amp;', 'and')
    s = s.replace('sq.', 'square')
    s = s.replace('gal.', 'gallon')
    
    return s

def preprocess(s):
    s = filter(filter_characters, s)
    s = s.lower()
    s = sanitize(s)
    
    return pattern.sub('', s)

# sanitize training and test    
train.loc[:, 'product_title'] = train.product_title.map(preprocess)
train.loc[:, 'search_term'] = train.search_term.map(preprocess)

test.loc[:, 'product_title'] = test.product_title.map(preprocess)
test.loc[:, 'search_term'] = test.search_term.map(preprocess)

# feature engineering
def query_title_overlap(row):
    query = row['search_term']
    title = row['product_title']
    query_words = query.split()
    
    count_overlap = 0
    for word in query_words:
        if query in title:
            count_overlap += 1
    
    return count_overlap

def compute_one_edit_distance(row):
    query = row['search_term']
    title = row['product_title']
    
    return 1 - seq_matcher(None, query, title).ratio()

train.loc[:, 'num_words_in_query'] = train.search_term.map(lambda x: len(x.split()))
test.loc[:, 'num_words_in_query'] = test.search_term.map(lambda x: len(x.split()))

train.loc[:, 'query_title_overlap'] = train.apply(query_title_overlap, axis=1)
test.loc[:, 'query_title_overlap'] = test.apply(query_title_overlap, axis=1)

train.loc[:, 'one_edit_distance'] = train.apply(compute_one_edit_distance, axis=1)
test.loc[:, 'one_edit_distance'] = test.apply(compute_one_edit_distance, axis=1)


    