#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:29:07 2022
"""
import os

import pandas as pd
from sklearn.metrics import ndcg_score

from src import config
from src.features.utils.features import PRODUCT_FEATURES, TOP_N
from src.features import item_most_popular
from src.features import folds


DATA_FOLDER = os.path.join(
    config.PRJ_DIR,
    'data/processed/ranks/item_most_popular/')

os.makedirs(DATA_FOLDER, exist_ok=True)


def process():
    """
    """
    item_most_pop_data = item_most_popular.get().set_index(
        'fecha_dato'
    )[PRODUCT_FEATURES]

    ndcg_scores = []
    for fold_id in folds.list():
        if config.VERBOSE:
            print('Eval Fold {0}'.format(fold_id))
        _, test, _ = folds.get(fold_id)

        true_relevance = test[PRODUCT_FEATURES]
        scores = item_most_pop_data.loc[test['fecha_dato']][PRODUCT_FEATURES]

        true_relevance_2 = true_relevance.drop('ind_cco_fin_ult1', axis=1)
        scores_2 = scores.drop('ind_cco_fin_ult1', axis=1)

        rank_path = os.path.join(DATA_FOLDER, '{0}.csv.gz').format(
            '{:02d}'.format(fold_id))

        scores.to_csv(rank_path, compression='gzip')

        ndcg_scores.append({
            'all_prod': ndcg_score(
                true_relevance,
                scores,
                k=TOP_N),
            'prods': ndcg_score(
                true_relevance_2,
                scores_2,
                k=TOP_N)
        })

    ndcg_scores = pd.DataFrame(ndcg_scores)

    ndcg_scores.to_csv(
        os.path.join(DATA_FOLDER, 'ndcg.csv'))
