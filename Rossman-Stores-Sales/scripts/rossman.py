import pandas as pd
from math import sqrt
import numpy as np
from sklearn.cross_validation import train_test_split

class Rossman():
    train_file_path = ''
    test_file_path = ''
    stores_file_path = ''

    def __init__(self, train_file_path, test_file_path, stores_file_path):
        """
        Sets in the file path for training, test and store
        info csv's
        """

        self.train_df = self.load_dataset(train_file_path, date_col='Date')
        self.test_df = self.load_dataset(test_file_path, date_col='Date')
        self.stores_df = self.load_dataset(stores_file_path)


    def load_dataset(self, file_path, date_col=None):
        """
        Loads dataset based on file path
        """

        if date_col:
            return pd.read_csv(file_path, parse_dates=[date_col])
        else:
            return pd.read_csv(file_path)


    def non_zero_sales_data(self):
        mask = self.train_df.Sales > 0
        return self.train_df[mask]

    def split_train_test_mask(self, test_size=0.3, random_state=0):

        """
        Splits the train_df into training and testing set
        training data will have all the examples except for last 6 weeks
        test data will examples for last 6 weeks
        """
        train_idx, test_idx = train_test_split(xrange(self.train_df.shape[0]), test_size=test_size, random_state=random_state)
        mask = np.ones(self.train_df.shape[0], dtype='int')
        mask[train_idx] = 1
        mask[test_idx] = 0

        mask = (mask==1)

        return mask

    def merge_stores_data(self):
        """
        Merge store information with training data and test data
        """

        self.train_df = pd.merge(self.train_df, self.stores_df, on='Store', how='left')
        self.test_df = pd.merge(self.test_df, self.stores_df, on='Store', how='left')


    def rmspe(self, y_true, y_pred):
        """
        Root Mean Square Percentage Error

        Args:
        y_true: true values for y
        y_pred: estimated values for y

        Returns: rmspe
        """

        n = len(y_true)
        e = []
        for i in range(n):
            if y_true[i] != 0:
                e.append((y_true[i] - y_pred[i]) / np.float(y_true[i]))
            else:
                e.append(0.0)

        e = np.array(e)
        e_squared = e ** 2

        return sqrt(np.sum(e_squared) / n)
