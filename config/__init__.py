import os
import abc
import inspect
import logging
import functools
from types import SimpleNamespace
from typing import Any
from collections import defaultdict

import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from hydra.utils import get_method, instantiate

import games
from networks.bases import DynamicsNet, PredictionNet, RepresentationNet

__all__ = ["config", "populate_config", "save_source_code"]

log = logging.getLogger(__name__)

# global configuration container
config = SimpleNamespace()


def populate_config(hydra_cfg: DictConfig):
    C = config
    to_cont = functools.partial(OmegaConf.to_container, resolve=True)

    # verify config schema by touching all values:
    OmegaConf.to_container(hydra_cfg, throw_on_missing=True)

    # GAME namespace
    C.game = SimpleNamespace()
    C.game.instance = instantiate(hydra_cfg.game.instance)
    assert isinstance(C.game.instance, games.bases.Game)
    C.game.calculate_reward = get_method(hydra_cfg.game.calculate_reward)

    # MCTS namespace
    C.mcts = SimpleNamespace(**hydra_cfg.mcts)
    C.mcts.get_node_action = get_method(hydra_cfg.mcts.get_node_action)
    C.mcts.get_node_target_policy = get_method(hydra_cfg.mcts.get_node_target_policy)
    C.mcts.get_node_selection_score = get_method(hydra_cfg.mcts.get_node_selection_score)

    # NETS namespace
    C.nets = SimpleNamespace()
    C.nets.initial_beliefs = torch.full(**to_cont(hydra_cfg.networks.initial_beliefs))
    C.nets.initial_latent_rep = torch.full(**to_cont(hydra_cfg.networks.initial_beliefs))

    C.nets.dynamics = instantiate(hydra_cfg.networks.dynamics)
    C.nets.prediction = instantiate(hydra_cfg.networks.prediction)
    C.nets.representation = instantiate(hydra_cfg.networks.representation)
    assert isinstance(C.nets.dynamics, DynamicsNet)
    assert isinstance(C.nets.prediction, PredictionNet)
    assert isinstance(C.nets.representation, RepresentationNet)

    # TRAIN namespace
    C.train = SimpleNamespace(**hydra_cfg.training)

    # PLAYER namespace
    from rl_player import RLPlayer  # this is here to break circular import

    C.player = SimpleNamespace()
    C.player.is_teammate = get_method(hydra_cfg.players.is_teammate)
    C.player.instances = tuple(map(instantiate, hydra_cfg.players.instances))
    msg = "There must be at least one RLPlayer involved to collect training data!"
    assert any(isinstance(p, RLPlayer) for p in C.player.instances), msg
    assert all(
        isinstance(p, games.bases.Player) or isinstance(p, RLPlayer)
        for p in C.player.instances
    )


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
            return

        namespace = "_".join(path[:-1])

        if inspect.isfunction(item):
            source = inspect.getsource(item)
        elif isinstance(item, torch.nn.Module):
            cls = item.__class__
            source = inspect.getsource(cls)
            blacklist = {
                cls,
                torch.nn.Module,
                abc.ABC,
                object,
                DynamicsNet,
                PredictionNet,
                RepresentationNet,
            }
            for superclass in filter(lambda c: c not in blacklist, cls.__mro__):
                sources[namespace].add(("superclass:", inspect.getsource(superclass)))
        else:
            return

        sources[namespace].add(("config path: " + ".".join(path), source))

    save_recursive(config, [])

    directory = "sources"
    os.mkdir(directory)
    n = 0
    for namespace, data in sources.items():
        with open(os.path.join(directory, f"{namespace}.py"), "w") as f:
            for explanation, sourcecode in sorted(data):
                f.write(f"# {explanation}\n{sourcecode}\n\n")
                n += 1
    log.info(f"Saved source code of {n} objects")
