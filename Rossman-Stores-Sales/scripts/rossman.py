import pandas as pd
from math import sqrt
import numpy as np
from sklearn.cross_validation import train_test_split

class Rossman():
    
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

    def split_train_test_mask(self, train_df, threshold_date, random_state=0):

        """
        Splits the train_df into training and testing set
        training data will have all the examples except for last 6 weeks
        test data will examples for last 6 weeks
        """
        features = train_df.columns.drop(['Customers', 'PromoInterval'])
        
        train_df_before_threshold = train_df[train_df.Date <= threshold_date][features]
        train_df_afer_threhold = train_df[train_df.Date > threshold_date][features]

        return train_df_before_threshold, train_df_afer_threhold



    def merge_stores_data(self):
        """
        Merge store information with training data and test data
        """

        self.train_df = pd.merge(self.train_df, self.stores_df, on='Store', how='left')
        self.test_df = pd.merge(self.test_df, self.stores_df, on='Store', how='left')


