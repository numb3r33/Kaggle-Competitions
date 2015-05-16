import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import cross_val_score
from sklearn.learning_curve import validation_curve

def load_data(path, index_col):
	"""
	Loads a csv file as pandas data frame
	"""
	return pd.read_csv(path, index_col=index_col)

def misclassification_percentage(y_true, y_pred):

	"""
	Returns misclassification percentage ( misclassified_examples / total_examples * 100.0)
	"""

	misclassified_examples = list(y_true == y_pred).count(False) * 1.
	total_examples = y_true.shape[0]
	return (misclassified_examples / total_examples) * 100.0

def validation_scores(model, X, y, n_iter=5, test_size=0.1):
	
	cv = ShuffleSplit(X.shape[0], n_iter=n_iter, test_size=test_size, random_state=0)
	test_scores = cross_val_score(model, X, y, cv=cv)

	return test_scores

def plot_validation_curves(param_values, train_scores, test_scores):
	for i in range(train_scores.shape[1]):
		plt.semilogx(param_values, train_scores[:, i], alpha=0.4, lw=2, c='b')
		plt.semilogx(param_values, test_scores[:, i], alpha=0.4, lw=2, c='g')
	
	plt.ylabel("score for LogisticRegression(fit_intercept=True)")
	plt.xlabel("C")
	plt.title('Validation curves for the C parameter');

def validation_curves(model, X, y, n_iter, test_size):
	n_Cs = 10
	Cs = np.logspace(-5, 5, n_Cs)
	cv = ShuffleSplit(X.shape[0], n_iter=n_iter, test_size=test_size, random_state=0)

	train_scores, test_scores = validation_curve(
		model, X, y, 'C', Cs, cv=cv)

	return (Cs, train_scores, test_scores)


class BaselineModel:

	"""
	Takes in the most majority class and number of training examples
	and returns its prediction as elements of majority class.
	e.g. In our current training set majority class is 0 then it would
	return all values as being 0 as our prediction.

	Any model that we develop must be compared with this baseline model.
	"""

	def __init__(self, majority_class, num_examples):
		self.majority_class = majority_class
		self.num_examples = num_examples

	def predict(self):
		return np.asarray([self.majority_class] * self.num_examples)

class Submission:
	"""
	Creates a submission in Kaggle competition format
	Column 1 will contain Patient Id and Column 2 will be 
	the prediction.
	"""
	def __init__(self, prediction):
		self.prediction = prediction

	def create_submission(self, path):
		with open(path, 'wb') as outfile:
			for (patient_id, pred) in enumerate(self.prediction):
				outfile.write(str(patient_id) + ',' + str(pred))
				outfile.write('\n')
			outfile.close()