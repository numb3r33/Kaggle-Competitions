import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
from HTMLParser import HTMLParser
from sklearn.cross_validation import StratifiedShuffleSplit
from bs4 import BeautifulSoup

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()
    
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

def load_file(filename, index_col):
    return pd.read_csv(filename, index_col=index_col).fillna('')

def prepareText(df):
    return list(df.apply(lambda x: '%s %s %s' %(x['query'], x['product_title'], x['product_description']), axis=1))

def how_uncorrelated(ytrue, model1pred, model2pred):
    count = 0

    for i in range(len(ytrue)):
        if ytrue[i] != model1pred[i] and ytrue[i] != model2pred[i]:
            if model1pred[i] != model2pred[i]:
                count += 1

    return (count * 1. / len(ytrue)) * 100.0

def strip_html(data):
    return [strip_tags(text) for text in data ]

def parseHTML(data):
    return ' '.join([p.get_text() for p in BeautifulSoup(data)])

def stem_text(data):
    stemmer = PorterStemmer()
    stemmed_text = []

    for text in data:
        words = text.split(' ')
        stemmed_words = []

        for word in words:
            stemmed_words.append(stemmer.stem(word.lower()))

        stemmed_text.append(' '.join(stemmed_words))

    return stemmed_text

def ssSplit(y, train_size=1000, random_state=0):
    sss = StratifiedShuffleSplit(y, 3, train_size=train_size, random_state=random_state)
    train_index, test_index = next(iter(sss))

    return (train_index, test_index) 


'''
Gets data for a particular relevance score
'''
def getText(data, y, label):
    return [data[i] for i in range(len(y)) if y[i] == label]


def lemmatize_text(data):
    lmtzr = WordNetLemmatizer()
    lemmatized_text = []

    for text in data:
        words = text.split(' ')
        lemmatized_words = []

        for word in words:
            lemmatized_words.append(lmtzr.lemmatize(word.lower()))

        lemmatized_text.append(' '.join(lemmatized_words))

    return lemmatized_text


def tweak_text(train):
    s_data = []
    stemmer = PorterStemmer()

    for i in range(train.shape[0]):
        s = (" ").join(["q"+ z for z in BeautifulSoup(train["query"].iloc[i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title.iloc[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description.iloc[i]).get_text(" ")
        s = re.sub("[^a-zA-Z0-9]"," ", s)
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s_data.append(s)
    
    return s_data

def lemmatize_text(train):
    s_data = []
    lmtzr = WordNetLemmatizer()
    
    for i in range(train.shape[0]):
        s = (" ").join(["q"+ z for z in BeautifulSoup(train["query"].iloc[i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title.iloc[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description.iloc[i]).get_text(" ")
        s = re.sub("[^a-zA-Z0-9]"," ", s)
        s = (" ").join([lmtzr.lemmatize(z) for z in s.split(" ")])
        s_data.append(s)
    
    return s_data


'''
Make a submission file in submissions folder in
current working directory that can be uploaded to
Kaggle.
'''

def make_submission(idx, preds, filename):
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("./submissions/" + filename, index=False)

