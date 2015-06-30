from sklearn.base import BaseEstimator
from nltk.corpus import stopwords
import numpy as np
from scipy import sparse
from nltk.stem import PorterStemmer


stop = stopwords.words('english')

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

def keyword_counter(document):
	query_in_resp_feat = document.apply(query_in_response, axis=1)
	num_query_feat = document.apply(num_query_in_response, axis=1)
	#jaccard_dist = document.apply(jaccard, axis=1)

	return np.array([query_in_resp_feat, num_query_feat]).T


def stack(features):
	features = sparse.hstack(features).tocsr()
	return features

def concat_examples(examples):
	total = sparse.vstack(examples).tocsr()
	return total
