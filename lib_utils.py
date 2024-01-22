import pathlib
import h5py
import logging
import rich.live
import rich.table

LOGGER = logging.getLogger(__name__)

class LiveTable:
    def __init__(self, *columns, table_kwargs=None, **kwargs):
        if table_kwargs is None:
            table_kwargs = {}
        
        self._columns = columns
        self._table = rich.table.Table(*columns, **kwargs)
        self._live_ctx = rich.live.Live(
            self._table, 
            refresh_per_second=1,
            **table_kwargs, 
            # auto_refresh=False,
        )
        self._auto_refresh = True

    def __enter__(self):
        self._live_ctx.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self._live_ctx.refresh()
        return self._live_ctx.__exit__(*args, **kwargs)

    def add_row(self, **column_dict):
        self._live_ctx.console.print("Adding row")
        assert len(column_dict) == len(self._columns), str(
            (column_dict.keys(), self._columns))
        reordered_column_dict = [column_dict[key] for key in self._columns]
        self._table.add_row(*reordered_column_dict)
        self._live_ctx.update(self._table, refresh=not self._auto_refresh)


def logging_basic_config(level=logging.DEBUG):
    logging.basicConfig(
        level=level,
        format="[%(asctime)s %(levelname)s %(module)s.%(funcName)s] %(message)s",
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