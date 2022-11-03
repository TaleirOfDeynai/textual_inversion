import numpy as np
import pytorch_lightning as pl
from typing import Any, Literal, Optional, TypeAlias, Union
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from torch.utils.data import DataLoader, get_worker_info
from functools import partial

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import default, instantiate_from_config, get_obj_from_str


# Copy from one config path to another when the first is unset and the second exists.
CfgDefFrom = tuple[Literal["default_from"], str, str]
# When the second path exists, always copy the value to the first path.
CfgSetFrom = tuple[Literal["set_from"], str, str]
# Set a config path to a given value if it is unset; the value must exist.
CfgDefAs = tuple[Literal["default_as"], str, Any]
# Always set a config path to a given value; the value must exist.
CfgSetAs = tuple[Literal["set_as"], str, Any]
# Ensures a config path no longer exists.
CfgDrop = tuple[Literal["drop"], str]
CfgOps = Union[CfgDefFrom, CfgSetFrom, CfgDefAs, CfgSetAs]

def operate_on_config(config: DictConfig, *args: CfgOps):
    """
    Performs some common operations on an OmegaConf `DictConfig`.  These
    are performed in the order given.  See the `CfgOps` type for operations
    that can be performed.

    The meaning of a value "existing" is its key must exist (when a path)
    and/or the value must not be `None`.  This allows `kwargs` defaults
    to work properly.
    """

    from omegaconf._utils import split_key

    # A reference to use as a default, to detect actually missing keys.
    NOT_EXISTS = {}

    # Why does this dumb library not have a built-in for this?
    def is_set(path: str):
        return OmegaConf.select(config, path, default=NOT_EXISTS) is not NOT_EXISTS

    def try_set(path: str, val: Any):
        if val is None: return
        OmegaConf.update(config, path, val)

    for op in args:
        if op[0] == "default_from":
            if is_set(op[1]): continue
            try_set(op[1], OmegaConf.select(config, op[2], default=None))
        elif op[0] == "set_from":
            try_set(op[1], OmegaConf.select(config, op[2], default=None))
        elif op[0] == "default_as":
            if is_set(op[1]): continue
            try_set(op[1], op[2])
        elif op[0] == "set_as":
            try_set(op[1], op[2])
        elif op[0] == "drop":
            keys = split_key(op[1])
            last_key = keys.pop()
            keys = "[" + "][".join(keys) + "]"
            sub_config = OmegaConf.select(config, keys, default=None)
            if sub_config is None: continue
            sub_config.pop(last_key, None)
        else:
            raise TypeError(f"Unknown config operation: {op}")
    return config


def worker_init_fn(_):
    worker_info = get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@dataclass
class ConfigInstantiable:
    """
    Data class to work with instantiable configuration a little easier.

    * `path` - The YAML path to the location.
    * `klasses` - A list of expected classes; use `None` to support anything.
    """
    path: str
    klasses: Optional[list[str]]

    @property
    def target(self):
        return f"{self.path}.target"
    @property
    def params(self):
        return f"{self.path}.params"

    def trim(self, init_path: str):
        """
        Returns a version of this object with `init_path` removed from the
        start of `path`, focusing it on a specific branch of the config.

        Work with dot notation paths only.
        """
        if init_path == self.path:
            # Wants to work with a config directly.
            return ConfigInstantiable(path="", klasses=self.klasses)
        rem_path = "{init_path}." if not init_path.endswith(".") else init_path
        assert self.path.startswith(rem_path), f"`{self.path}` does not start with `{init_path}`."
        return ConfigInstantiable(path=self.path[len(rem_path):], klasses=self.klasses)
    
    def direct(self):
        """Returns a version of this object with an empty `path`."""
        return self.trim(self.path)

    def supports(self, klass):
        """Checks if the given `klass` is one of the supported `klasses`."""
        if self.klasses is None: return True
        return klass in self.klasses

    def ctor(self, config: DictConfig):
        """
        Attempts to obtain the target constructor.  Will return `None` when
        the configuration is missing.
        """
        klass_config = OmegaConf.select(config, self.path, default=None)
        if klass_config is None: return None
        klass_target = OmegaConf.select(klass_config, "target", default="")
        assert self.supports(klass_target), f"Will not obtain the target; `{klass_target}` is not one of: {self.klasses}"
        return get_obj_from_str(klass_target)

    def instantiate(self, config: DictConfig, **kwargs):
        """
        Attempts to instantiate the object from the given config.  Will return
        `None` when the configuration is missing.
        """
        klass_config = OmegaConf.select(config, self.path, default=None)
        if klass_config is None: return None
        klass_target = OmegaConf.select(klass_config, "target", default="")
        assert self.supports(klass_target), f"Cannot instantiate; `{klass_target}` is not one of: {self.klasses}"
        return instantiate_from_config(klass_config, **kwargs)


DatasetKey = Literal["train", "validation", "test", "predict"]


class BasicDataModule(pl.LightningDataModule):
    """
    Basic data module.

    Provide a static `normalize_config` method in inheriting classes to
    operate on the config prior to any instantiation.  This method has
    the following signature:

    `(config: DictConfig, opt: Namespace) -> None
    """
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs: dict[DatasetKey, Any] = dict()
        self.num_workers = default(num_workers, batch_size * 2)
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        # this call should only happen once, so we can quickly report
        # some stats about the dataset that may be distributed as we
        # instantiate (and presumably cache) them
        print("#### Data #####")
        for k in self.dataset_configs:
            dataset = self.instantiate_data(k)
            print(f"{k}, {dataset.__class__.__name__}, {len(dataset)}")

    def setup(self, stage=None):
        self.datasets = dict((k, self.instantiate_data(k)) for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
    
    def instantiate_data(self, config_key: DatasetKey):
        """
        Instantiates the data indicated by `config_key`.

        Intended to be overridden with custom logic.
        """
        return instantiate_from_config(self.dataset_configs[config_key])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets["train"], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets["validation"], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets["train"], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets["predict"], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)