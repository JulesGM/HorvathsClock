import fire
import pathlib
import sklearn.impute
import h5py

import collections

import lib_utils
import logging

LOGGER = logging.getLogger(__name__)

def main(path, mode):
    # Logger basic config
    lib_utils.logging_basic_config()
    data = lib_utils.load_split_data(path)

    LOGGER.debug("Imputing")
    imputer = sklearn.impute.SimpleImputer(strategy=mode)
    LOGGER.debug("Training imputer")
    imputer.fit(data["train"]["features"])
    LOGGER.debug("Imputing train")
    data["train"]["features"] = imputer.transform(data["train"]["features"])
    LOGGER.debug("Imputing validation")
    data["validation"]["features"] = imputer.transform(data["validation"]["features"])
    LOGGER.debug("Imputing test")
    data["test"]["features"] = imputer.transform(data["test"]["features"])

    lib_utils.write_split_data(path.with_suffix(f".imputed-{mode}.h5"), data)

if __name__ == "__main__":
    fire.Fire(main)