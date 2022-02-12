# -*- coding: utf-8 -*-
import os

import pandas as pd
import datetime

from src import config
from src.features import train_ver2
from src.features.utils.features import PRODUCT_FEATURES


DATA_PATH = os.path.join(
    config.PRJ_DIR,
    'data/processed/most_pop_item_at_{0}_days.csv')

# The most popular item ranking given a date is conditioned to
# a time window starting from the target date and up to the number of days
# configured in the following constants.
DAYS_TIME_WINDOW = [60, 90, 120]


def process():
    """
    """
    data = train_ver2.get()

    dates = data['fecha_dato'].unique()

    for days_time_w in DAYS_TIME_WINDOW:
        item_most_pop = []
        for date_upper in dates:
            date_lower = pd.to_datetime(
                date_upper) - datetime.timedelta(days=days_time_w)

            # Subset valid data within the time window.
            date_data = data[(
                data['fecha_dato'] < date_upper
            ) & (
                data['fecha_dato'] >= date_lower
            )]

            # Get product frequency
            product_freq = date_data[PRODUCT_FEATURES].sum()

            # Normalize product frequency
            rank = (product_freq / product_freq.sum()).fillna(0)

            rank['fecha_dato'] = date_upper
            item_most_pop.append(rank)

        item_most_pop_df = pd.DataFrame(item_most_pop)
        data_path = DATA_PATH.format(days_time_w)

        if config.VERBOSE:
            print(data_path)
        item_most_pop_df.to_csv(data_path, index=False)


def get(days=60):
    """
    """
    data = pd.read_csv(DATA_PATH.format(days))

    return data


if __name__ == '__main__':
    """
    """
    process()
