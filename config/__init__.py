import os
import abc
import inspect
import logging
import functools
from types import SimpleNamespace
from typing import Any, Callable
from collections import defaultdict

import hydra
import torch
import omegaconf
from omegaconf import OmegaConf, DictConfig, ListConfig

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


OConfig = DictConfig | ListConfig


def traverse_config(cfg: OConfig, callback: Callable[[OConfig, str], None]) -> None:
    """
    Traverse a config depth-first and call a function at each node that has str keys.
    """
    if isinstance(cfg, DictConfig):
        for key in cfg:
            if not OmegaConf.is_missing(cfg, key):
                traverse_config(cfg[key], callback)

            assert type(key) is str
            callback(cfg, key)

    elif isinstance(cfg, ListConfig):
        for item in cfg:
            traverse_config(item, callback)


def merge_structured_config_defaults(cfg: OConfig) -> None:
    """
    This function takes an OmegaConf Config and recursively merges the default values of
    the underlying structured config classes in-place.

    This is because Omegaconf disabled auto-expanding of nested structured configs:
    https://github.com/omry/omegaconf/issues/412
    The proposed solutions are unnecessary verbose (default assignments) and worse, they
    don't allow for default values to propagate into variable-length lists.
    """

    def merge_defaults(cfg: OConfig, key: str) -> None:
        t = OmegaConf.get_type(cfg, key)
        if omegaconf._utils.is_structured_config(t):
            d = OmegaConf.to_container(OmegaConf.structured(t))
            with omegaconf.read_write(cfg):
                OmegaConf.update(cfg, key, d)

    traverse_config(cfg, merge_defaults)


def populate_config(cfg: DictConfig) -> None:
    # we want the _partial_ keys in our config for the instantiate call later
    merge_structured_config_defaults(cfg)

    # verify config schema by touching all values:
    OmegaConf.to_container(cfg, throw_on_missing=True)

    # transform all configs with _target_ keys into their class instances (or partials)
    cfg = hydra.utils.instantiate(cfg)  # type: DictConfig # type: ignore [no-redef]

    def ensure_callable(cfg: DictConfig, key: str) -> None:
        val = cfg[key]
        if not callable(val):
            cfg[key] = hydra.utils.get_method(val)
        # otherwise callable class (already instatiated)

    ensure_callable(cfg.game, "reward_fn")
    ensure_callable(cfg.mcts, "node_action_fn")
    ensure_callable(cfg.mcts, "node_target_policy_fn")
    ensure_callable(cfg.mcts, "node_selection_score_fn")
    ensure_callable(cfg.players, "is_teammate_fn")

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
