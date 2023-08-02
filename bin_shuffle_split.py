import collections
import logging

import h5py
import pathlib
import fire
import numpy as np

import lib_utils

LOGGER = logging.getLogger(__name__)

def main(path):
    
    path = pathlib.Path(path)
    assert path.exists(), f"File `{path}` does not exist"

    LOGGER.debug(f"Loading data: {path}")
    with h5py.File(path, "r") as fin:
        data = {k: fin[k][:] for k in fin}
    
    LOGGER.debug("Transposing")
    data["features"] = np.transpose(data["features"])
    NB_PATIENTS = data["features"].shape[0]
    assert all([len(x) == NB_PATIENTS for x in data.values()]), (
        f"Not all data have the same number of patients: {[(k, len(v)) for k, v in data.items()]}"
    )

    LOGGER.debug("Shuffling the data. This will take a while.")
    new_indices = np.random.permutation(NB_PATIENTS)
    data = {k: v[new_indices] for k, v in data.items()}

    split_data = collections.defaultdict(dict)
    LOGGER.debug("Split into train, validation, and test. 80/10/10")
    LOGGER.debug("Doing Train")
    split_data["train"] = {k: v[:int(NB_PATIENTS * 0.8)] for k, v in data.items()}
    LOGGER.debug("Doing Validation")
    split_data["validation"] = {k: v[int(NB_PATIENTS * 0.8):int(NB_PATIENTS * 0.9)] for k, v in data.items()}
    LOGGER.debug("Doing Test")
    split_data["test"] = {k: v[int(NB_PATIENTS * 0.9):] for k, v in data.items()}

    lib_utils.write_split_data(path.with_suffix(".shuffled_and_split.h5"), split_data)

    LOGGER.debug(f"All done. Have a nice day.")

if __name__ == "__main__":
    fire.Fire(main)