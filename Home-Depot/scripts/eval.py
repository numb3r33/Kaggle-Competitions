import numpy as np

from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_squared_error

def eval_model(models, X, y):
	'''
	Takes in an array of models , fit each of the model on the training examples given
	Calculates 5-fold cross validation
	'''
	cv = ShuffleSplit(len(y), n_iter=5, test_size=0.3)

	scores = []
	for train, test in cv:
		scores_combined = np.zeros(len(test))

		for clf in models:
			X_train, y_train = X.iloc[train], y.iloc[train]
			X_test, y_test = X.iloc[test], y.iloc[test]
			clf.fit(X_train, y_train)
			relevance = clf.predict(X_test)
			print("score: %f" % np.sqrt(mean_squared_error(y_test, relevance)))
			scores_combined += relevance

		scores_combined /= len(models) * 1.
		scores.append(np.sqrt(mean_squared_error(y_test, scores_combined)))
		print("combined score: %f" % scores[-1])

	return (np.mean(scores), np.std(scores))