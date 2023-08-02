import pathlib
import h5py
import logging
LOGGER = logging.getLogger(__name__)


def logging_basic_config(level=logging.DEBUG):
    logging.basicConfig(
        level=level,
        format="[%(asctime)s %(levelname)s %(module)s.%(funcname)s] %(message)s",
    )

def load_split_data(path):
    path = pathlib.Path(path)
    assert path.exists(), f"File `{path}` does not exist"

    LOGGER.debug(f"Loading data: {path}")
    with h5py.File(path, "r") as fin:
        data = {}
        for split in fin:
            data[split] = {}
            for information_type in fin[split]:
                key = f"{split}/{information_type}"
                LOGGER.debug(f"Doing {key}")
                data[split][information_type] = fin[key][:]

    return data

def write_split_data(new_path, data):
    LOGGER.debug(f"Writing to file: {new_path}")
    with h5py.File(new_path, "w") as fout:
        for split in data:
            assert split in ["train", "validation", "test"], (
                f"Split `{split}` not in ['train', 'validation', 'test']"
            )
            for information_type in data[split]:
                key = f"{split}/{information_type}"
                LOGGER.debug(f"Doing {key}")
                fout.create_dataset(
                    key,
                    data=data[split][information_type]
                )