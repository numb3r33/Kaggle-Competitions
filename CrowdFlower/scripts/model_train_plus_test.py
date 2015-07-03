from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression


def TFIDF(Xtrain, Xwhole):
	tfv = TfidfVectorizer(min_df=3, max_df=600, max_features=None,
		  strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
		  ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')

	if Xwhole == None:
		return (tfv.fit_transform(Xtrain), tfv)
	else:
		tfv.fit(Xwhole)
		return (tfv.transform(Xtrain), tfv)

def build_non_linear_model(Xtrain, y):
	
	svd = TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
	scl = StandardScaler(copy=True, with_mean=True, with_std=True)
	
	Xtrain = svd.fit_transform(Xtrain)
	Xtrain = scl.fit_transform(Xtrain)

	clf = SVC(C=10.0, kernel='rbf', degree=3, 
		  gamma=0.0, coef0=0.0, shrinking=True, probability=False, 
		  tol=0.001, cache_size=200, class_weight=None, 
		  verbose=False, max_iter=-1, random_state=None)


	return (clf.fit(Xtrain, y), svd, scl)

def build_linear_model(X, y):
	select = SelectPercentile(score_func=chi2, percentile=20)
	clf = SVC(C=10.0, kernel='linear', probability=True)

	X = select.fit_transform(X, y)
	return (clf.fit(X, y), select)

def build_knn_model(Xtrain, y):
	svd = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
	scl = StandardScaler(copy=True, with_mean=True, with_std=True)
	
	Xtrain = svd.fit_transform(Xtrain)
	Xtrain = scl.fit_transform(Xtrain)

	clf = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute')

	return (clf.fit(Xtrain, y), svd, scl)


def make_predictions(model, options, Xtest):
	if options.has_key('tfv'):
		Xtest = options['tfv'].transform(Xtest)
	
	if options.has_key('svd'):
		Xtest = options['svd'].transform(Xtest)
	
	if options.has_key('scl'):
		Xtest = options['scl'].transform(Xtest)

	if options.has_key('select'):
		Xtest = options['select'].transform(Xtest)

	return model.predict(Xtest)