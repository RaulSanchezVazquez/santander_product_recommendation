#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

from scipy import sparse

from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k

from src import config
from src.features.utils.features import PRODUCT_FEATURES, TOP_N
from src.features import user_most_popular
from src.features import folds


DATA_FOLDER = os.path.join(
    config.PRJ_DIR,
    'data/processed/ranks/fact_matrix_2/')

os.makedirs(DATA_FOLDER, exist_ok=True)


ALPHA = 1e-5
LR = 0.01
EPOCHS = 120
N_COMPONENTS = 10
EARLY_STOPPING_ROUNDS = 10
VAL_SIZE = .05


def get_ids(entities):
    """
    """
    idx_to_ent, ent_to_idx = {}, {}
    for ent_it, ent in enumerate(entities):
        idx_to_ent[ent_it] = ent
        ent_to_idx[ent] = ent_it

    return idx_to_ent, ent_to_idx


def get_val_mask(train_user_items, val_size=.3):
    """
    """

    train_cardinality = (
        train_user_items.shape[0] * train_user_items.shape[1])

    val_mask = np.full(
        train_cardinality, False)

    n_val = int(val_size * train_cardinality)
    val_mask[:n_val] = True

    np.random.shuffle(val_mask)

    val_mask = val_mask.reshape(
        train_user_items.shape)

    train_mask = ~val_mask

    return train_mask.astype(int), val_mask.astype(int)


def process():
    """
    """

    idx_to_i, i_to_idx = get_ids(PRODUCT_FEATURES)

    # Get user centroids (last 90 days window)
    user_most_pop_df = user_most_popular.get()

    ndcg_scores = []
    for fold_id in folds.list():
        if config.VERBOSE:
            print('Eval Fold {0}'.format(fold_id))
        train, test, _ = folds.get(fold_id)

        # Train only with latets user interaction
        train = train.sort_values(
            'fecha_dato'
        ).drop_duplicates(
            'ncodpers',
            keep='last')

        # Get user-id encoding objects
        idx_to_u, u_to_idx = get_ids(
            train['ncodpers'].unique())

        user_ids = train['ncodpers'].apply(lambda x: u_to_idx[x])

        user_items = user_most_pop_df.loc[train['id']]
        user_items = (user_items[PRODUCT_FEATURES] > 0).astype(int)
        user_items.index = user_ids
        user_items.sort_index(inplace=True)

        is_cold_start = user_items.sum(axis=1) == 0
        cold_start_u = is_cold_start[is_cold_start].index[0]

        train_mask, val_mask = get_val_mask(
            user_items, val_size=VAL_SIZE)

        val_user_items = sparse.csr_matrix(
            user_items.values * val_mask)

        train_user_items = sparse.csr_matrix(
            user_items.values * train_mask)

        recsys_model = LightFM(
            no_components=N_COMPONENTS,
            learning_rate=LR,
            loss='warp',
            learning_schedule='adagrad',
            user_alpha=ALPHA,
            item_alpha=ALPHA)

        metric = []
        for epoch in range(EPOCHS):
            recsys_model.fit_partial(
                train_user_items,
                epochs=1,
                num_threads=config.N_JOBS)

            epoch_auc = auc_score(
                recsys_model,
                val_user_items,
                num_threads=config.N_JOBS)

            auc = epoch_auc.mean()

            precision = precision_at_k(
                recsys_model,
                val_user_items,
                num_threads=8).mean()

            metric.append(auc)

            print(f"{epoch} {auc} {precision}")

            if len(metric) > EARLY_STOPPING_ROUNDS:
                current = metric[-1]
                previous = metric[-EARLY_STOPPING_ROUNDS]

                if current < previous:
                    break

        test_user_ids = test['ncodpers'].apply(
            lambda x: u_to_idx[x] if x in u_to_idx else cold_start_u
        ).values

        n_items = len(PRODUCT_FEATURES)
        test_users, test_items = [], []
        for user_id in test_user_ids:
            test_users += [user_id] * n_items
            test_items += range(0, n_items)

        scores = pd.DataFrame(
            recsys_model.predict(
                user_ids=np.array(test_users).astype(int),
                item_ids=np.array(test_items).astype(int),
                num_threads=config.N_JOBS
            ).reshape(-1, n_items),
            columns=PRODUCT_FEATURES)

        scores.to_csv(
            os.path.join(
                DATA_FOLDER,
                '{0}.csv.gz'
            ).format('{:02d}'.format(fold_id)),
            compression='gzip')

        ndcg_all_prod = ndcg_score(
            test[PRODUCT_FEATURES].values,
            scores.values,
            k=TOP_N)

        ndcg_prods = ndcg_score(
            test[PRODUCT_FEATURES].drop('ind_cco_fin_ult1', axis=1).values,
            scores.drop('ind_cco_fin_ult1',axis=1).values,
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
