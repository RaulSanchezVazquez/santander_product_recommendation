# -*- coding: utf-8 -*-
import os
import math

import glob
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

from src import config
from src.data.utils import parallel
from src.features import train_ver2
from src.features.utils.features import PRODUCT_FEATURES


MOST_POP_USER_FOLDER = os.path.join(
    config.PRJ_DIR,
    'data/processed/most_pop_user/')

os.makedirs(MOST_POP_USER_FOLDER, exist_ok=True)


DAYS_TIME_WINDOW = 90

_U_DATA_GRP = None
_BATCH_SIZE = 100


def process():
    """
    """
    global _U_DATA_GRP

    data = train_ver2.get()
    _U_DATA_GRP = data.groupby('ncodpers')

    user_ids = get_users_ids(data)

    # Split user ids into batches
    n_batches = math.ceil(len(user_ids) / _BATCH_SIZE)
    user_id_batches = np.array_split(
        user_ids,
        n_batches)

    pbar = tqdm(total=n_batches)
    for batch_it, batch in enumerate(user_id_batches):
        parallel.apply(
            get_user_most_pop,
            batch)
        pbar.update(1)


def get_users_ids(data):
    """
    """

    user_most_pop_filepaths = glob.glob(
        os.path.join(MOST_POP_USER_FOLDER, '*'))

    user_ids_already_processed = set([
        int(user_path.split('/')[-1].replace('.csv.gz', ''))
        for user_path in user_most_pop_filepaths])

    all_user_ids = set(data['ncodpers'].unique())
    user_ids = list(all_user_ids - user_ids_already_processed)

    return user_ids


def get_user_most_pop(user_id):
    """
    """
    global _U_DATA_GRP

    # Get user data
    user_data = _U_DATA_GRP.get_group(
        user_id
    ).set_index('fecha_dato').sort_index()

    user_most_pop = []
    for date_upper in user_data.index:
        date_lower = date_upper - datetime.timedelta(days=DAYS_TIME_WINDOW)

        date_user_data = user_data[(
            user_data.index < date_upper
        ) & (
            user_data.index >= date_lower
        )]

        rank = date_user_data[
            PRODUCT_FEATURES
        ].sum()

        rank = (rank / rank.sum()).fillna(0)

        user_most_pop.append(rank)

    user_most_pop = pd.DataFrame(user_most_pop)
    user_most_pop['ncodpers'] = user_id
    user_most_pop['fecha_dato'] = user_data.index

    user_most_pop_filepath = os.path.join(
        MOST_POP_USER_FOLDER,
        '{0}.csv.gz'.format(user_id))

    user_most_pop.to_csv(
        user_most_pop_filepath,
        compression='gzip')

    return


if __name__ == '__main__':
    """
    """
    process()
