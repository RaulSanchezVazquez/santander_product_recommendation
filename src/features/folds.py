# -*- coding: utf-8 -*-
import os
import glob

import numpy as np
import pandas as pd

from src import config
from src.data.utils import parallel
from src.features import train_ver2
from src.features.utils.features import PRODUCT_FEATURES, TOP_N


DATA_FOLDER = os.path.join(
    config.PRJ_DIR,
    'data/interim/folds')

os.makedirs(DATA_FOLDER, exist_ok=True)


TRAIN_WINDOW_SIZE = 6


def get_experiment_folds(data):
    """Get folds.

    Mapping of fold-id to the data partition.
    Each partition contains dates that comprise the partition fold.

    Returns
    -------
    folds : dict
        The folds and their partitions.
    """
    dates = data['fecha_dato'].unique()

    folds = {}
    fold_id = 0
    for it, date in enumerate(dates):
        if it < TRAIN_WINDOW_SIZE:
            continue

        train_end = it
        train_start = train_end - TRAIN_WINDOW_SIZE

        fold_id += 1
        folds[fold_id] = {
            'train': dates[train_start:train_end],
            'test': [date]}

    return folds


def process():
    """Split train dataset in folds.
    """

    data = train_ver2.get()

    folds = get_experiment_folds(data)

    data.set_index('fecha_dato', inplace=True)
    for fold_id, fold in folds.items():

        fold_path = os.path.join(
            DATA_FOLDER,
            '{:02d}'.format(fold_id))
        os.makedirs(fold_path, exist_ok=True)

        for partition_name, partition in fold.items():
            if partition_name != 'test':
                continue

            partition_data = data[
                data.index.isin(pd.to_datetime(partition))]

            partition_data['id'] = (
                partition_data['ncodpers'].astype(str)
            ) + '-' + (
                partition_data.index.astype(str))

            parition_path = os.path.join(
                fold_path,
                '{0}.csv.gz'.format(partition_name))

            if partition_name == 'test':
                user_prod_sum = partition_data[
                    PRODUCT_FEATURES].sum(axis=1)
                partition_data = partition_data[user_prod_sum > 0]

            if config.VERBOSE:
                print(parition_path)
            partition_data.to_csv(parition_path, compression='gzip')

            if partition_name == 'test':
                rank_stats_path = os.path.join(
                    fold_path,
                    '{0}_rank_stats.csv.gz'.format(partition_name))

                rank_stats = get_ranks_and_idcg(partition_data)

                rank_stats.to_csv(rank_stats_path)


def get(fold_id=1):
    """Get fold partitions.
    """
    fold_path = os.path.join(
        DATA_FOLDER,
        '{:02d}'.format(fold_id))

    train = pd.read_csv(
        os.path.join(fold_path, 'train.csv.gz'))
    test = pd.read_csv(
        os.path.join(fold_path, 'test.csv.gz'))
    rank_stats = pd.read_csv(
        os.path.join(fold_path, 'test_rank_stats.csv.gz'))

    return train, test, rank_stats


def list():
    """List folds.
    """
    fold_list = sorted([
        int(x.split('/')[-1])
        for x in glob.glob(os.path.join(DATA_FOLDER, '*'))])

    return fold_list


def get_ranks_and_idcg(data):
    """Get rank ground-truth and IDCG.
    """

    idcg_top_n = 1 / np.log2(np.arange(2, TOP_N + 1))

    products = data[PRODUCT_FEATURES]

    ranks = parallel.apply(
        _get_prods,
        products.iterrows())

    rank_stats = pd.DataFrame({
        'id': data['id'],
        'rank': ranks,
        'idcg': [idcg_top_n[:len(x)].sum() for x in ranks]
    }).set_index('id')

    return rank_stats


def _get_prods(params):
    """Get products.
    """
    _, products = params
    products = products[products > 0].index.tolist()

    return products


if __name__ == '__main__':
    """
    """
    process()
