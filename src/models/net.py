#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:29:07 2022
"""
import os
from collections import OrderedDict
import math

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

from src import config
from src.features.utils.features import PRODUCT_FEATURES, TOP_N
from src.features import user_most_popular
from src.features import folds


DATA_FOLDER = os.path.join(
    config.PRJ_DIR,
    'data/processed/ranks/net2/')

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


class Net(nn.Module):
    """NN model for dense and sparse features.
    """
    def __init__(self,
         n_items,
         n_dense,
         emb_size,
         out_size,
         embedding_dropout=0.02,
         dropout=0.2):

        super().__init__()

        self.emb_drop = nn.Dropout(embedding_dropout)
        self.I = nn.Embedding(n_items, emb_size)

        self.hidden = nn.Sequential(OrderedDict([
            ('layer_1', nn.Linear(emb_size + n_dense, int(emb_size / 2))),
            ('relu_1', nn.ReLU()),
            ('droput_1', nn.Dropout(dropout)),
            ('layer_2', nn.Linear(int(emb_size / 2), int(emb_size / 2))),
            ('relu_2', nn.ReLU()),
            ('dropout_2', nn.Dropout(dropout)),
            ('layer_out', nn.Linear(int(emb_size / 2), out_size))
        ]))


        def init(seq_item):
            if type(seq_item) == nn.Linear:
                torch.nn.init.xavier_uniform_(seq_item.weight)
                seq_item.bias.data.fill_(0.01)

        self.I.weight.data.uniform_(-0.05, 0.05)
        self.hidden.apply(init)


    def forward(self, X_emb, X_dense):
        """Forward pass.

        Parameters
        ----------
        X_emb : torch.tensor
            The hist-item index.
        X_dense : torch.tensort
            The dense features.
        """

        X_emb_ = self.I(X_emb)
        X_emb_ = self.emb_drop(X_emb_)

        X = torch.cat([
            X_emb_,
            X_dense],
            axis=1).float()

        scores = self.hidden(X)

        return scores


def get_batches(X_train_str, batch_size=512):
    """Split data indexes in randomized batches.
    """
    n = len(X_train_str.index)

    n_batches = math.floor(n / batch_size)

    random_choice = np.random.choice(
        X_train_str.index,
        size=n,
        replace=False)

    batches = random_choice[: n_batches * batch_size].reshape(
        n_batches,
        batch_size)

    return batches


def id_to_x(user_most_pop_df, ids):
    """Encode user-most popular encoding to indicator variables.
    """

    X = user_most_pop_df.loc[ids]

    X_bool = (
        X[PRODUCT_FEATURES] > 0
    ).astype(int)

    X_str = X_bool.apply(
        lambda x: str(x.values.tolist()),
        axis=1)

    return X_str


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

        scaler = get_normalizer(train)

        val = train.sample(
            int(train.shape[0] * .3),
            random_state=42)

        train = train[~train.index.isin(val.index)]

        y_train, y_val, y_test = (
            train[PRODUCT_FEATURES],
            val[PRODUCT_FEATURES],
            test[PRODUCT_FEATURES])

        X_train_str, X_val_str, X_test_str = (
            id_to_x(user_most_pop_df, train.index),
            id_to_x(user_most_pop_df, val.index),
            id_to_x(user_most_pop_df, test.index))

        items = list(set(
            X_train_str.unique()
        ).union(
            X_val_str.unique()
        ).union(
            X_test_str.unique()))
        idx_to_items, items_to_idx, n_items = get_item_ids(items)

        X_val_emb = torch.tensor(
            [items_to_idx[x] for x in X_val_str])
        X_test_emb = torch.tensor(
            [items_to_idx[x] for x in X_test_str])

        X_train_dense = normalize(train, scaler, user_most_pop_df)
        X_val_dense = torch.tensor(
            normalize(val, scaler, user_most_pop_df).values)
        X_test_dense = torch.tensor(
            normalize(test, scaler, user_most_pop_df).values)

        model = Net(
            n_items=n_items,
            n_dense=X_val_dense.shape[1],
            emb_size=int(2**8),
            out_size=len(PRODUCT_FEATURES))

        loss = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=3e-1)

        val_ndcg = []

        for epoch in range(N_EPOCHS):
            for batch in get_batches(X_train_str, batch_size=int(2**11)):
                X_dense = X_train_dense.loc[batch]
                X_dense = torch.tensor(X_dense.values)

                X_emb = torch.tensor(
                    [items_to_idx[x] for x in X_train_str.loc[batch]])

                y_tensor = torch.tensor(
                    y_train.loc[batch].values.astype(float))

                model.train()
                # All items
                if epoch < 3:
                    scores = model(X_emb, X_dense)
                    output = loss(scores, y_tensor)

                    optimizer.zero_grad()
                    output.backward()
                    optimizer.step()

                # Exclude most pop item
                scores = model(X_emb, X_dense)
                idx_most_pop = PRODUCT_FEATURES.index('ind_cco_fin_ult1')

                scores = torch.cat([
                    scores[:, :idx_most_pop],
                    scores[:, idx_most_pop + 1:]],
                    axis=1)
                y_tensor = torch.cat([
                    y_tensor[:, :idx_most_pop],
                    y_tensor[:, idx_most_pop + 1:]],
                    axis=1)
                output = loss(scores, y_tensor)

                optimizer.zero_grad()
                output.backward()
                optimizer.step()

            model.eval()

            val_scores = model(
                X_val_emb, X_val_dense)

            val_scores = pd.DataFrame(
                nn.Sigmoid()(val_scores).detach().numpy(),
                columns=y_val.columns)

            epoch_ndcg_score = ndcg_score(
                y_val.drop(
                    'ind_cco_fin_ult1',
                    axis=1),
                val_scores.drop(
                    'ind_cco_fin_ult1',
                    axis=1),
                k=TOP_N)

            val_ndcg.append(epoch_ndcg_score)

            if config.VERBOSE:
                print(val_ndcg[-1])

            if len(val_ndcg) > EARLY_STOPPING_ROUNDS:
                current = val_ndcg[-1]
                previous = val_ndcg[-EARLY_STOPPING_ROUNDS]

                if current < previous:
                    break

        test_scores = model(
            X_test_emb, X_test_dense)
        test_scores = pd.DataFrame(
            nn.Sigmoid()(test_scores).detach().numpy(),
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
