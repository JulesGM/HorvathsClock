#!/usr/bin/env python
# coding: utf-8



import collections
import math
import pathlib
import io
import zipfile

import h5py
import fire
import more_itertools
import pandas as pd
import numpy as np
import pyreadr
import rich
import rich.markup
import rich.table
from IPython.display import display
from tqdm.notebook import tqdm
import sklearn.ensemble
import sklearn.linear_model
import sklearn.impute
import sklearn.metrics

N_PATIENTS = 8375 - 1
N_FEATURES = 485515 - 3


def prep_h5_file(input_file_name="age_methylation_v1.txt", output_file_name="output.h5"):
    input_file_name = pathlib.Path(input_file_name)
    output_file_name = pathlib.Path(output_file_name)

    assert input_file_name.exists(), f"Input file `{input_file_name}` does not exist"

    with h5py.File(output_file_name, "w") as f:
        f.create_dataset("age", dtype="f", shape=(N_PATIENTS,))
        f.create_dataset("features", dtype="f", shape=(N_FEATURES, N_PATIENTS))
        f.create_dataset("tissue", dtype=h5py.string_dtype(encoding='utf-8'), shape=(N_PATIENTS,))
        f.create_dataset("sample_id", dtype=h5py.string_dtype(encoding='utf-8'), shape=(N_PATIENTS,))

        for i, line in tqdm(enumerate(open(input_file_name, "r")), total=N_FEATURES):
            if i == 0:
                f["sample_id"][:] = line.strip().split("\t")[1:]
            elif i == 1:
                f["age"][:] = [float(x) for x in line.strip().split("\t")[1:]]
            elif i == 2:
                f["tissue"][:] = line.strip().split("\t")[1:]
            else:
                f["features"][i - 3] = [float(x) if x != "NA" else float("nan") for x in line.strip().split("\t")[1:]]


def build_data(input_file_name="output.h5"):
    print("Loading raw data")
    with h5py.File(input_file_name, "r") as f:
        all_ = f["features"][:]
        ages = f["age"][:]
    
    print("Transposing")
    all_ = all_.transpose()

    # permutations = np.random.permutation(all_.shape[0])
    # # This takes forever
    # all_ = all_[permutations]
    # ages = ages[permutations]

    print("Splitting")
    train_x = all_[:int(all_.shape[0] * 0.8)]
    test_x = all_[int(all_.shape[0] * 0.8):]
    del all_
    train_y = ages[:train_x.shape[0]]
    test_y = ages[train_x.shape[0]:]

    print("Done")
    return train_x, train_y, test_x, test_y


def fit_hgbr(train_x, train_y, test_x, test_y):


    model = sklearn.ensemble.HistGradientBoostingRegressor()
    model.fit(train_x, train_y)

    print(model.score(test_x, test_y))


def fit_bag(*, train_x, train_y, test_x, test_y, n_features, n_samples, n_estimators, do_imputation, estimator):
    if do_imputation:
        print("Imputation: fit")
        imputer = sklearn.impute.SimpleImputer()
        imputer.fit(train_x)
        
        print("Imputation: transform train_x")
        train_x = imputer.transform(train_x)
        
        print("Imputation: transform test_x")
        test_x = imputer.transform(test_x)
    
    model = sklearn.ensemble.BaggingRegressor(
        max_features=n_features, 
        max_samples=n_samples, 
        n_estimators=n_estimators, 
        estimator=estimator,
    )
    print("Fitting.")
    model.fit(train_x, train_y)

    print("Score:")
    if do_imputation:
        print("Mean absolute error:")
        print(f"{sklearn.metrics.mean_absolute_error(test_x, test_y) = }")
        print("Mean squared error:")
        print(f"{sklearn.metrics.mean_squared_error(test_x, test_y) = }")
    print("R2 score:")
    print(f"{model.score(test_x, test_y) = }")

    return model



def fit_lasso(train_x, train_y, test_x, test_y):
    model = sklearn.linear_model.Lasso()
    
    print("Imputation: fit")
    imputer = sklearn.impute.SimpleImputer()
    imputer.fit(train_x)
    
    print("Imputation: transform train_x")
    train_x = imputer.transform(train_x)
    
    print("Imputation: transform test_x")
    test_x = imputer.transform(test_x)
    
    print("Lasso fit")
    model.fit(train_x, train_y)

    print("Score")
    try:
        print(model.score(test_x, test_y))
    except ValueError:
        print("ValueError", ValueError)
    
    print("Done")
    return model

def main():
    train_x, train_y, test_x, test_y = build_data()
    fit_bag(
        do_imputation=False,
        n_estimators=12,
        n_features=1.,
        n_samples=1.,
        train_x=train_x, 
        train_y=train_y, 
        test_x=test_x, 
        test_y=test_y, 
        n_jobs=4,
        estimator=sklearn.ensemble.HistGradientBoostingRegressor(),
    )

if __name__ == "__main__":
    fire.Fire()