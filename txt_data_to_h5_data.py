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

if __name__ == "__main__":
    fire.Fire(prep_h5_file)
