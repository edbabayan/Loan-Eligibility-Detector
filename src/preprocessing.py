import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.data_analysis import DataAnalysis


class Preprocessor:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.numerical_columns = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.train_df.select_dtypes(exclude=[np.number]).columns.tolist()

    def preprocess_dataframes(self):
        self.merge_incomes()
        self.encode_categorical_columns()
        self.process_numeric_columns()
        return self.train_df, self.test_df

    def merge_incomes(self):
        self.train_df['Income'] = self.train_df['ApplicantIncome'] + self.train_df['CoapplicantIncome']
        self.test_df['Income'] = self.test_df['ApplicantIncome'] + self.test_df['CoapplicantIncome']
        self.train_df.drop(columns=['ApplicantIncome', 'CoapplicantIncome'], inplace=True)
        self.test_df.drop(columns=['ApplicantIncome', 'CoapplicantIncome'], inplace=True)

    def encode_categorical_columns(self):
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.train_df[col] = le.fit_transform(self.train_df[col])
            self.test_df[col] = le.transform(self.test_df[col])

    def process_numeric_columns(self):
        # log transformation
        self.train_df[self.numerical_columns] = np.log(self.train_df[self.numerical_columns])
        self.test_df[self.numerical_columns] = np.log(self.test_df[self.numerical_columns])

        # scaling
        min_max = MinMaxScaler()
        self.train_df = min_max.fit_transform(self.train_df)
        self.test_df = min_max.transform(self.test_df)



if __name__ == '__main__':
    from src.config import CFG
    analyzer = DataAnalysis(CFG.data_dir.joinpath('train.csv'), CFG.data_dir.joinpath('test.csv'),
                                                                  'Loan_Status')
    cleaned_train, cleaned_test = analyzer.get_cleaned_dataframes()

    processor = Preprocessor(cleaned_train, cleaned_test)
    processor.merge_incomes()
