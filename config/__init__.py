import os
import abc
import inspect
import logging
import functools
from types import SimpleNamespace
from typing import Any
from collections import defaultdict

import torch
from omegaconf import OmegaConf, DictConfig

from config.schema import BaseConfig
from networks.bases import (
    DynamicsNet,
    NetworkBase,
    PredictionNet,
    RepresentationNet,
)

__all__ = ["C", "populate_config", "save_source_code"]

log = logging.getLogger(__name__)

# Global configuration container
# It is empty at first and later filled in-place by populate_config()
# So every other module can conveniently import it directly
C = BaseConfig.placeholder()


def populate_config(cfg: DictConfig) -> None:
    # verify config schema by touching all values:
    OmegaConf.to_container(cfg, throw_on_missing=True)

    msg = "There must be at least one RLPlayer involved to collect training data!"


def save_source_code():
    # to yield reproducable experiments, save the source code of all functions and
    # networks referenced in the config, even additional ones not designated in the config
    # schema

    # {"config namespace": {("config path / origin", "source code")}}
    sources = defaultdict(set)  # type: defaultdict[set[tuple[str, str]]]

    def save_recursive(item: Any, path: list[str]):
        if isinstance(item, SimpleNamespace):
            for name, child in vars(item).items():
                save_recursive(child, path + [name])

        namespace = "_".join(path[:-1])

        if isinstance(item, functools.partial):
            item = item()  # partials are probably factories, instantiate them

        if inspect.isfunction(item) and path[-1] != "optimizer_factory":
            source = inspect.getsource(item)
        elif isinstance(item, NetworkBase):
            cls = item.__class__
            source = inspect.getsource(cls)
            blacklist = {
                cls,
                torch.nn.Module,
                abc.ABC,
                object,
                NetworkBase,
                DynamicsNet,
                PredictionNet,
                RepresentationNet,
            }
            for superclass in filter(lambda c: c not in blacklist, cls.__mro__):
                sources[namespace].add(("superclass:", inspect.getsource(superclass)))
        else:
            return

        sources[namespace].add(("config path: " + ".".join(path), source))

    save_recursive(C, [])

    directory = "sources"
    os.mkdir(directory)
    n = 0
    for namespace, data in sources.items():
        with open(os.path.join(directory, f"{namespace}.py"), "w") as f:
            for explanation, sourcecode in sorted(data):
                f.write(f"# {explanation}\n{sourcecode}\n\n")
                n += 1
    log.info(f"Saved source code of {n} objects")
