#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:29:07 2022
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from src import config
from src.features.utils.features import PRODUCT_FEATURES, TOP_N
from src.features import user_most_popular
from src.features import folds


DATA_FOLDER = os.path.join(
    config.PRJ_DIR,
    'data/processed/ranks/net_multilabel/')

NDCG_PATH = os.path.join(DATA_FOLDER, 'ndcg.csv')

os.makedirs(DATA_FOLDER, exist_ok=True)

# Dense features
DENSE_FEATURES = ['age', 'renta']

# Training params
N_EPOCHS = 30
EARLY_STOPPING_ROUNDS = 3

def get_item_ids(items):
    """Get mappings of items to indices.
    """
    n_items = len(items)
    idx_to_items, items_to_idx = {}, {}
    for item_idx, item in enumerate(items):
        idx_to_items[item_idx] = item
        items_to_idx[item] = item_idx

    return idx_to_items, items_to_idx, n_items


def get_normalizer(data):
    """Get normalizer object.
    """
    dense_features = data[DENSE_FEATURES].fillna(0).apply(np.log1p)

    scaler = StandardScaler()
    scaler.fit(dense_features)

    return scaler


def normalize(data, scaler, user_most_pop_df):
    """Normalize dense features.
    """

    data_log = data[DENSE_FEATURES].astype(
        float
    ).fillna(0).apply(np.log1p).fillna(0)

    data_norm = scaler.transform(data_log)

    prd_features = user_most_pop_df.loc[data.index][PRODUCT_FEATURES]

    dense_f = np.concatenate([
        data_norm,
        prd_features.values
    ], axis=1)

    dense_f = pd.DataFrame(dense_f, index=data.index)

    return dense_f


def process():
    """Fit model.
    """

    user_most_pop_df = user_most_popular.get()

    if not os.path.exists(NDCG_PATH):
        ndcg_scores = []
    else:
        ndcg_scores = pd.read_csv(
            NDCG_PATH
        ).drop(
            'Unnamed: 0',
            axis=1
        ).apply(
            lambda x: x.to_dict(),
            axis=1
        ).tolist()

    for fold_id in folds.list()[len(ndcg_scores):]:
        if config.VERBOSE:
            print('Eval Fold {0}'.format(fold_id))
        train, test, _ = folds.get(fold_id)
        train.set_index('id', inplace=True)
        test.set_index('id', inplace=True)

        train = train.sample(int(train.shape[0] * .3))

        y_train, y_test = (
            train[PRODUCT_FEATURES],
            test[PRODUCT_FEATURES])

        scaler = get_normalizer(train)

        X_train_dense = normalize(
            train, scaler, user_most_pop_df)
        X_test_dense = normalize(
            test, scaler, user_most_pop_df)

        model = MLPClassifier()
        model.fit(
            X_train_dense.values,
            y_train)

        test_scores = model.predict_proba(X_test_dense.values)
        test_scores = pd.DataFrame(
            test_scores,
            columns=y_test.columns)

        rank_path = os.path.join(DATA_FOLDER, '{0}.csv.gz').format(
            '{:02d}'.format(fold_id))
        test_scores.to_csv(rank_path, compression='gzip')

        ndcg_all_prod = ndcg_score(
            y_test,
            test_scores,
            k=TOP_N)

        ndcg_prods = ndcg_score(
            y_test.drop('ind_cco_fin_ult1', axis=1),
            test_scores.drop('ind_cco_fin_ult1',axis=1),
            k=TOP_N)

        test_ndcg = {
            'all_prod': ndcg_all_prod,
            'prods': ndcg_prods}

        ndcg_scores.append(test_ndcg)

        if config.VERBOSE:
            print('----')
            print(ndcg_scores[-1])
            print('----')

    ndcg_scores_df = pd.DataFrame(ndcg_scores)
    ndcg_scores_df.to_csv(
        os.path.join(DATA_FOLDER, 'ndcg.csv'))
