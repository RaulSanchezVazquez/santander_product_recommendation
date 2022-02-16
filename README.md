Santander Product Recommendation Challenge
==========================================

Setup
------
Download the training dataset from [kaggle](https://www.kaggle.com/c/santander-product-recommendation/data) and store it under the following folder:
```
data/externa/train_ver2.csv
```

Setup and environment with conda using the `environment.yml`:
```
$ conda env create -f environment.yml
```

Activate your environment as:
```
$ conda activate recsys
```

The Code
=========
The code is structured following the guidelines from [cookie-cutter template for data science](https://github.com/drivendata/cookiecutter-data-science).

To reproduce the results for this challenger follow the instructions in the next section.

The slide presentation of the solution can be found under:

```
docs/docs/Recsys Challenge.pdf
```

Data Cleaning and Folds
------------------------

Filter some rows from the original dataset:
```
$ ipython src/features/train_ver2.py
```

Split the dataset in folds, where each fold consists of a train and test partitions:
```
$ ipython src/features/folds.py
```

Compute Most Popular Datasets
------------------------------
Compute most popular items per date (item most popular):
```
$ ipython src/features/item_most_popular.py
```

Compute most popular items by user and date (user most popular):
```
$ ipython src/features/user_most_popular.py
```
Running `src/features/user_most_popular.py` may take several hours. This script starts a lazy process, which means that if the execution of the script is interrupted and restarted, the script will retake the computation where it left the last time.

Train the Models
-------------
Train and evaluate the item most popular model:
```
$ ipython src/features/item_most_popular.py
```
Train and evaluate the user most popular model:
```
$ ipython src/models/user_most_popular.py
```
Train and evaluate the RecSys Net (item embeddings + dense features):
```
$ ipython src/features/recsysnet.py
```
Train a Matrix Factorization (interaction features only):
```
$ ipython src/features/lightFM.py
```
Train a Multi-label Multilayer-Perceptron (dense features only):
```
$ ipython src/features/net_multilabel.py
```

Evaluation
------------
Run a jupyter notebook in the top-most folder of the project and run the following notebooks to get the
evaluation plots:
```
notebooks/Dataset Analysis.ipynb             
```

```
notebooks/Model evaluations.ipynb
```

```
notebooks/Model evaluations Cold Star.ipynb  
```
