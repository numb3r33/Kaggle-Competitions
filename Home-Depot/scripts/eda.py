# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 08:58:07 2016

@author: abhishek
"""
import pandas as pd
import re

# import libraries used for nlp
from __future__ import division
from nltk import word_tokenize
from nltk import FreqDist

# load train, test, description and attributes files
train = pd.read_csv('./data/train.csv', index_col='id')
test = pd.read_csv('./data/test.csv', index_col='id')

description = pd.read_csv('./data/product_descriptions.csv')
attributes = pd.read_csv('./data/attributes.csv')


## Frequency Analysis
def default_tokenizer(sentence):
    return sentence.split(' ')

def tokenize(sentence, tokenizer_type='word'):
    if tokenizer_type == 'word':
        return word_tokenize(sentence)
    else:
        return default_tokenizer(sentence)

def tokenize_sentences(sentences, n):
    tokens = []
    
    for i in range(0, n):
        tokens.extend(tokenize(sentences[i]))
    
    return tokens

def frequency_analysis(search_terms, n=50, num_terms=5):
    tokens_list = tokenize_sentences(search_terms, n=n)
    fdist = FreqDist(tokens_list)
    
    return fdist.most_common(n=num_terms)
    

## Pattern Matching
query_list = [(idx, w) for (idx, w) in enumerate(train.search_term.values) if re.search(r'ft.', w)]
relevance_scores = [train.iloc[idx]['relevance'] for (idx, w) in query_list]