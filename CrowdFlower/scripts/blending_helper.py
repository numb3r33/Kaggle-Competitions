from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

def prepareTrainData(X, Xwhole):
	tfv = TfidfVectorizer(min_df=3, max_features=None,
		  strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
		  ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')

	if Xwhole == None:
		X = tfv.fit_transform(X)
	else:
		tfv.fit(Xwhole)
		X = tfv.transform(X)

	svd = TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
	scl = StandardScaler(copy=True, with_mean=True, with_std=True)
	
	X = svd.fit_transform(X)
	X = scl.fit_transform(X)

	return (X, tfv, svd, scl)

def prepareTestData(Xtest, tfv, svd, scl):
	Xtest = tfv.transform(Xtest)
	Xtest = svd.transform(Xtest)
	Xtest = scl.transform(Xtest)

	return Xtest
