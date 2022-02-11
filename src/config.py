#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

# Project path
PRJ_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-2])

# Path to store data
DATA_DIR = os.path.join(PRJ_DIR, 'data')

# Flag used to silence prints
VERBOSE = True

# Number of jobs
N_JOBS = 6

# Create project folders
_prj_folders = [
    'data',
    'data/raw',
    'data/processed',
    'data/interim',
    'data/external',
    'models',
    'reports'
]


for folder in _prj_folders:
    os.makedirs(
        os.path.join(PRJ_DIR, folder),
        exist_ok=True)
