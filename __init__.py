from src.config import CFG

with open(CFG.root.joinpath("VERSION")) as f:
    __version__ = f.read().strip()
