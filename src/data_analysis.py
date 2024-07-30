import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.config import CFG

raw_train = pd.read_csv(CFG.data_dir.joinpath('train.csv'))
raw_test = pd.read_csv(CFG.data_dir.joinpath('test.csv'))


class DataAnalysis:
    def __init__(self, train_csv, test_csv, prediction_column):
        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)
        self.prediction_column = prediction_column

        # Split target variable from train dataset
        self.train_y = self.train_df[self.prediction_column].copy()
        self.train_df.drop(columns=[self.prediction_column], inplace=True)

        # Drop unnecessary columns
        self.train_df.drop(columns=['Loan_ID'], inplace=True)
        self.test_df.drop(columns=['Loan_ID'], inplace=True)

        # Store column names for imputations
        self.numerical_columns = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.train_df.select_dtypes(exclude=[np.number]).columns.tolist()

    def get_info(self, dataset='train'):
        """Return the info of the dataframe (for train and test separately)."""
        if dataset == 'train':
            return self.train_df.info()
        elif dataset == 'test':
            return self.test_df.info()
        else:
            raise ValueError("dataset must be 'train' or 'test'")

    def check_duplicates(self, dataset='train'):
        """Check for duplicates (for train and test separately)."""
        if dataset == 'train':
            return self.train_df[self.train_df.duplicated()]
        elif dataset == 'test':
            return self.test_df[self.test_df.duplicated()]
        else:
            raise ValueError("dataset must be 'train' or 'test'")

    def drop_duplicates(self):
        """Drop duplicates in both train and test datasets."""
        self.train_df.drop_duplicates(inplace=True)
        self.test_df.drop_duplicates(inplace=True)

    def get_columns(self):
        """Return the numeric and categorical columns names of the dataframe."""
        return self.numerical_columns, self.categorical_columns

    def impute_missing_values(self):
        """Perform imputations on numeric and categorical columns."""
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_imputer.fit(self.train_df[self.categorical_columns])

        self.train_df[self.categorical_columns] = cat_imputer.transform(self.train_df[self.categorical_columns])
        self.test_df[self.categorical_columns] = cat_imputer.transform(self.test_df[self.categorical_columns])

        num_imputer = SimpleImputer()
        num_imputer.fit(self.train_df[self.numerical_columns])

        self.train_df[self.numerical_columns] = num_imputer.transform(self.train_df[self.numerical_columns])
        self.test_df[self.numerical_columns] = num_imputer.transform(self.test_df[self.numerical_columns])

    def print_missing_values(self, dataset='train'):
        """Print the count of missing values in each column (for train and test separately)."""
        if dataset == 'train':
            print(self.train_df.isna().sum())
        elif dataset == 'test':
            print(self.test_df.isna().sum())
        else:
            raise ValueError("dataset must be 'train' or 'test'")

    def get_cleaned_dataframes(self):
        """Return the cleaned train and test dataframes."""
        self.drop_duplicates()
        self.impute_missing_values()
        return self.train_df, self.test_df, self.train_y


if __name__ == '__main__':
    analysis = DataAnalysis(CFG.data_dir.joinpath('train.csv'), CFG.data_dir.joinpath('test.csv'),
                                                                  'Loan_Status')
    cleaned_train, cleaned_test, y_train = analysis.get_cleaned_dataframes()
    print('')