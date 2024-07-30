from pathlib import Path


class CFG:
    root = Path(__file__).parent.parent.absolute()
    data_dir = root.joinpath('data')
    preprocessing_dir = root.joinpath('preprocessed')

    trained_models = root.joinpath('trained_models')
