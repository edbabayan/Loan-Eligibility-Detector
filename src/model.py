import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class LoanabilityDetector:
    def __init__(self, test_size=0.3):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.test_size = test_size
        self.model = LogisticRegression()

    def train(self, train_df, y_train):
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_dataset(train_df, y_train)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        prediction = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, prediction)
        return acc

    def split_dataset(self, train_df, y):
        X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=self.test_size,
                                                            random_state=0)
        return X_train, X_test, y_train, y_test

    def predict(self, applicant_info):
        prediction = self.model.predict(applicant_info)
        return prediction


if __name__ == '__main__':
    from src.config import CFG
    from src.data_analysis import DataAnalysis
    from src.preprocessing import Preprocessor

    analyzer = DataAnalysis(CFG.data_dir.joinpath('train.csv'), CFG.data_dir.joinpath('test.csv'),
                                                                  'Loan_Status')
    cleaned_train, cleaned_test, _y_train = analyzer.get_cleaned_dataframes()

    processor = Preprocessor(cleaned_train, cleaned_test)
    _train_df, _test_df = processor.preprocess_dataframes()

    _model = LoanabilityDetector()
    _model.train(_train_df, _y_train)
    print(_model.evaluate())

    CFG.trained_models_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(_model, CFG.trained_models_dir.joinpath('loanability_detector.pkl'))
