from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


def build_linear_model(X, y):
	tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

	select = SelectPercentile(score_func=chi2, percentile=20)
	clf = SVC(C=25.0, kernel='linear')

	X = tfv.fit_transform(X)
	X = select.fit_transform(X, y)
	return (clf.fit(X, y), tfv, select)

def build_non_linear_model(X, y):
	tfv = TfidfVectorizer(min_df=3,  max_df=500, max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

	svd = TruncatedSVD(n_components=300)
	scl = StandardScaler()
	clf = SVC(C=10.0)

	tfv.fit(X)
	X = tfv.transform(X)
	X = svd.fit_transform(X)
	X = scl.fit_transform(X)

	return (clf.fit(X, y), tfv, svd, scl)

def build_knn_model(X, y):
	tfv = TfidfVectorizer(min_df=3,  max_df=500, max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

	svd = TruncatedSVD(n_components=300)

	clf = KNeighborsClassifier(n_neighbors=5)

	tfv.fit(X)
	X = tfv.transform(X)
	X = svd.fit_transform(X)
	
	return (clf.fit(X, y), tfv, svd)

def build_naive_bayes(X, y):
	tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

	clf = MultinomialNB(alpha=.01)

	X = tfv.fit_transform(X)
	
	return (clf.fit(X, y), tfv)	


def build_stopwords_tweak_model(X, y):
	tfv = TfidfVectorizer(min_df=3, max_df=500, max_features=None,
		  strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
		  ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')


	svd = TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
	scl = StandardScaler(copy=True, with_mean=True, with_std=True)
	clf = SVC(C=10.0, kernel='rbf', degree=3, 
		  gamma=0.0, coef0=0.0, shrinking=True, probability=False, 
		  tol=0.001, cache_size=200, class_weight=None, 
		  verbose=False, max_iter=-1, random_state=None)

	tfv.fit(X)
	X = tfv.transform(X)
	X = svd.fit_transform(X)
	X = scl.fit_transform(X)

	return (clf.fit(X, y), tfv, svd, scl)


'''
This function can be used for both linear kernel and SGDClassifier
'''
def linear_model_predictions(model, tfv, select, Xtest):
	Xtest = tfv.transform(Xtest)
	Xtest = select.transform(Xtest)

	return model.predict(Xtest)

def non_linear_model_predictions(model, tfv, svd, scl, Xtest):
	Xtest = tfv.transform(Xtest)
	Xtest = svd.transform(Xtest)
	Xtest = scl.transform(Xtest)

	return model.predict(Xtest)

def naive_bayes_predictions(model, tfv, Xtest):
	Xtest = tfv.transform(Xtest)
	return model.predict(Xtest)

def knn_model_predictions(model, tfv, svd, Xtest):
	Xtest = tfv.transform(Xtest)
	Xtest = svd.transform(Xtest)

	return model.predict(Xtest)