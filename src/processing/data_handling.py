import os
import sys

import joblib
import pandas as pd
from pathlib import Path

from src.config import CFG


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))


# Load the dataset
def load_dataset(file_name):
    filepath = CFG.data_dir.joinpath(file_name)
    _data = pd.read_csv(filepath)
    _data.drop("Loan_ID", axis=1, inplace=True)
    return _data


# Deserialization
def load_pipeline(pipeline_to_load):
    save_path = os.path.join(CFG.trained_models_dir, pipeline_to_load)
    model_loaded = joblib.load(save_path)
    print(f"Model has been loaded")
    return model_loaded


# Separate X and y
def separate_data(data):
    X = data.drop(CFG.TARGET, axis=1)
    y = data[CFG.TARGET]
    return X, y