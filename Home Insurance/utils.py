import pandas as pd
import numpy as np


def load_data(train_filename='./data/train.csv', test_filename='./data/test.csv'):
	
	print 'Loading datasets'
	
	train = pd.read_csv(train_filename, parse_dates=['Original_Quote_Date'], index_col='QuoteNumber')
	test = pd.read_csv(test_filename, parse_dates=['Original_Quote_Date'], index_col='QuoteNumber')

	print 'Setting Quote Number as index'

	return train, test


def prepare_sample(train, n=1000):
	features = train.columns.drop('QuoteConversion_Flag')

	train_2013 = train[train.Original_Quote_Date.dt.year==2013].sample(n=n)
	train_2014 = train[train.Original_Quote_Date.dt.year==2014].sample(n=n)
	train_2015 = train[train.Original_Quote_Date.dt.year==2015].sample(n=n)

	train_merged = pd.concat([train_2013, train_2014, train_2015], axis=0)
	train_merged_shuffle = train_merged.iloc[np.random.permutation(len(train_merged))]

	X = train_merged_shuffle[features]
	y = train_merged_shuffle['QuoteConversion_Flag']

	return X, y

def random_sample(train, n):
	features = train.columns.drop('QuoteConversion_Flag')

	train = train.take(np.random.permutation(len(train))[:n])

	X = train[features]
	y = train['QuoteConversion_Flag']

	return X, y
