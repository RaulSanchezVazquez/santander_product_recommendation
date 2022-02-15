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


DATA_FOLDER = os.path.join(
    config.PRJ_DIR,
    'data/processed/most_pop_user/')

DATA_PATH = os.path.join(
    config.PRJ_DIR,
    'data/processed/most_pop_user.csv.gz')

os.makedirs(DATA_FOLDER, exist_ok=True)

# User Time window
DAYS_TIME_WINDOW = 90

# Batch size for multi-process
_BATCH_SIZE = 100

# Cache variables
_U_DATA_GRP = None


def process():
    """Compute user most popular within the time window.
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
    """Get users with most popular items missings.
    """

    user_most_pop_filepaths = glob.glob(
        os.path.join(DATA_FOLDER, '*'))

    user_ids_already_processed = set([
        int(user_path.split('/')[-1].replace('.csv.gz', ''))
        for user_path in user_most_pop_filepaths])

    all_user_ids = set(data['ncodpers'].unique())
    user_ids = list(all_user_ids - user_ids_already_processed)

    return user_ids


def get_user_most_pop(user_id):
    """Process a single user most popular items.
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
        DATA_FOLDER,
        '{0}.csv.gz'.format(user_id))

    user_most_pop.to_csv(
        user_most_pop_filepath,
        compression='gzip')

    return


def get():
    """Get user most popular.
    """
    data = pd.read_csv(DATA_PATH).set_index('id')

    return data


def concat():
    """Merge all users most popular items.
    """

    user_files = glob.glob(os.path.join(DATA_FOLDER, '*'))
    n_batches = math.ceil(len(user_files) / _BATCH_SIZE)
    user_id_batches = np.array_split(
        user_files,
        n_batches)

    pbar = tqdm(total=n_batches)
    data = []
    for batch_it, batch in enumerate(user_id_batches):
        data += parallel.apply(
            pd.read_csv,
            batch)
        pbar.update(1)

    data = pd.concat(data)
    data.drop('Unnamed: 0', axis=1, inplace=True)
    data['id'] = (
        data['ncodpers'].astype(str)
    ) + '-' + (
        data['fecha_dato'])

    data.to_csv(DATA_PATH, compression='gzip')



if __name__ == '__main__':
    """
    """
    process()
    concat()
