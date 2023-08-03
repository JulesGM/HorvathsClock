import fire
import pathlib
import sklearn.impute
import h5py
import numpy as np

import collections

import lib_utils
import logging

LOGGER = logging.getLogger(__name__)

def main(path):
    # Logger basic config
    lib_utils.logging_basic_config()
    path = pathlib.Path(path)
    data = lib_utils.load_split_data(path)

    LOGGER.debug("Normalizing.")
    LOGGER.debug("Computing mean and std")
    mean = np.nanmean(data["train"]["features"], axis=0)
    std = np.nanstd(data["train"]["features"], axis=0)
    
    for split in data:
        LOGGER.debug(f"Normalizing {split}")
        data[split]["features"] -= mean
        data[split]["features"] /= std

    lib_utils.write_split_data(path.with_suffix(f".normalized.h5"), data)

if __name__ == "__main__":
    fire.Fire(main)