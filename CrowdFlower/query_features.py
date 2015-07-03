from sklearn.base import BaseEstimator
from nltk.corpus import stopwords
import numpy as np
from scipy import sparse
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn


stop = stopwords.words('english')

def is_query_in_response(train):
    query_terms = train['query'].split(' ')
    response = train['product_title'] + ' ' + train['product_description']
    
    stemmer = PorterStemmer()
    query_terms_stemmed = [stemmer.stem(q) for q in query_terms]
    response_stemmed = ''.join([stemmer.stem(r) for r in response])
    stop = stopwords.words('english')
       
    keyword = False
    
    for q in query_terms_stemmed:
        if q not in stop:
            keyword = True
            if response_stemmed.lower().find(q) == -1:
                return 0
    
    if keyword == False:
        return 0
    else:
        return 1



def query_in_response(doc):
	query_terms = doc['query'].split(' ')
	unique_terms = list(set(query_terms))
	response = doc['product_title'] + ' ' + doc['product_description']
	keyword = False
	
	for q in unique_terms:
		if q not in stop:
			keyword = True
			
			if response.lower().find(q) == -1:
				return 0

	if keyword == False:
		return 0
	else:
		return 1


def num_query_in_response(doc):
	query_terms = doc['query'].split(' ')
	unique_terms = list(set(query_terms))
	response = doc['product_title'] + ' ' + doc['product_description']
	count = 0

	for q in unique_terms:
		if q not in stop:
			if response.lower().find(q) == -1:
				count += 1

	return count

def query_synonymns_check(x):
    query = x['query'].lower()
    query_terms = list(set(query.split()))
    response = x['product_title'].lower() + ' ' + x['product_description'].lower()
    query_synonymns = []
    stop = stopwords.words('english')
    
    for q in query_terms:
        for i, j in enumerate(wn.synsets(q)):
            query_synonymns.extend(j.lemma_names)
    
    count = 0
    for qsynonym in query_synonymns:
        if qsynonym not in stop and response.find(qsynonym) != -1:
            count += 1
    
    return count



def jaccard(x):
    query = x['query'].lower()
    title = x['product_title'].lower()
    description = x['product_description'].lower()
    response = title + ' ' + description
    
    query_set = set(query.split(' '))
    response_set = set(response.split(' '))
    
    query_response_intersection_len = len(query_set & response_set)
    query_response_union_len = len(query_set | response_set)
    
    return (query_response_intersection_len * 1.) / (query_response_union_len)


def query_length(x):
	return len(x['query'].split(' '))


def keyword_counter(document):
	query_in_resp_feat = document.apply(query_in_response, axis=1)
	num_query_feat = document.apply(num_query_in_response, axis=1)
	# query_synonym_count_feat = document.apply(query_synonymns_check, axis=1)
	# query_length_feat = document.apply(query_length, axis=1)
	# jaccard_dist = document.apply(jaccard, axis=1)

	# return np.array([query_in_resp_feat, num_query_feat, query_synonym_count_feat]).T
	
	#query_in_resp_feat = document.apply(is_query_in_response, axis=1)

	return np.array([query_in_resp_feat, num_query_feat]).T

def stack(features):
	features = sparse.hstack(features).tocsr()
	return features

def concat_examples(examples):
	total = sparse.vstack(examples).tocsr()
	return total
