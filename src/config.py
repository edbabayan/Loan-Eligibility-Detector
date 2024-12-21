from pathlib import Path


class CFG:
    root = Path(__file__).parent.parent.absolute()
    data_dir = root.joinpath('data')
    preprocessing_dir = root.joinpath('preprocessed')

    TEST_FILE = "train.csv"

    trained_models_dir = root.joinpath('trained_models')

    MODEL_NAME = 'loanability_detector.pkl'

    TARGET = 'Loan_Status'

    FEATURE_TO_ADD = ['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

    DROP_FEATURES = ['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

    LOG_FEATURES = ['income_annum', 'loan_amount', 'total_assets_value']
