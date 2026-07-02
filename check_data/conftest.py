import pytest
import pandas as pd
from pathlib import Path

TRAIN_PATH = "data/fraud_train.csv"
TEST_PATH  = "data/fraud_test.csv"
V_COLS     = [f"V{i}" for i in range(1, 29)]


@pytest.fixture(scope="session")
def df_train():
    if not Path(TRAIN_PATH).exists():
        pytest.skip(f"{TRAIN_PATH} no existe — ejecuta las etapas 1-3 primero")
    return pd.read_csv(TRAIN_PATH)


@pytest.fixture(scope="session")
def df_test():
    if not Path(TEST_PATH).exists():
        pytest.skip(f"{TEST_PATH} no existe — ejecuta las etapas 1-3 primero")
    return pd.read_csv(TEST_PATH)
