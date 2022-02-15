# -*- coding: utf-8 -*-
import os
import time
import pandas as pd

from src import config


DATA_PATH = os.path.join(
    config.PRJ_DIR,
    'data/external/train_ver2.csv')


def get():
    """Fetch data.

    Train dataset.

    Return
    ------
    data : pandas.DataFrame.
        The data.
    """

    start = time.time()
    data = pd.read_csv(DATA_PATH)
    end = time.time()

    if config.VERBOSE:
        print('Read data in {0}mins'.format(
            round((end - start) / 60, 2)))

    return data
