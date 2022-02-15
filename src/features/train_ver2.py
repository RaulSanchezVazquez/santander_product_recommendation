# -*- coding: utf-8 -*-
import os
import time
import pandas as pd

from src import config
from src.data import train_ver2
from src.features.utils.features import PRODUCT_FEATURES

DATA_PATH = os.path.join(
    config.PRJ_DIR,
    'data/interim/train_ver2.csv')

DATE_COLUMNS = ["fecha_dato", "fecha_alta"]


def get():
    """Get train dataset.

    Some rows are dropped and data types are parsed.
    """
    start = time.time()

    # Read data
    data = train_ver2.get()

    # Parse data types
    data["age"] = pd.to_numeric(data["age"], errors="coerce")

    for int_col in PRODUCT_FEATURES:
        data[int_col] = pd.to_numeric(
            data[int_col],
            errors="coerce"
        ).fillna(0).astype(int)

    for date_column in DATE_COLUMNS:
        data[date_column] = pd.to_datetime(
            data[date_column], format="%Y-%m-%d")

    # Remove Age Outliers and Nans
    lower_q, upper_q = data['age'].quantile([.01, .99])
    data = data[(
        data['age'] > lower_q
    ) & (
        data['age'] < upper_q)]

    # Remove suspicious nans
    data = data[~data["ind_nuevo"].isnull()]

    end = time.time()

    if config.VERBOSE:
        print('Read data in {0}mins'.format(
            round((end - start) / 60, 2)))

    return data
