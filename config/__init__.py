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
from hydra.utils import get_method, instantiate

import games
from networks.bases import (
    Networks,
    DynamicsNet,
    NetworkBase,
    PredictionNet,
    RepresentationNet,
)

__all__ = ["C", "populate_config", "save_source_code"]

log = logging.getLogger(__name__)

# global configuration container
C = SimpleNamespace()


def populate_config(hydra_cfg: DictConfig):
    to_cont = functools.partial(OmegaConf.to_container, resolve=True)
    pinstantiate = functools.partial(instantiate, _partial_=True)

    def to_namespace_recurse(x) -> SimpleNamespace:
        if isinstance(x, dict):
            return SimpleNamespace(**{k: to_namespace_recurse(v) for k, v in x.items()})
        if isinstance(x, list):
            return list(map(to_namespace_recurse, x))
        return x

    # verify config schema by touching all values:
    OmegaConf.to_container(hydra_cfg, throw_on_missing=True)

    # GAME namespace
    C.game = to_namespace_recurse(to_cont(hydra_cfg.game))
    C.game.instance = instantiate(hydra_cfg.game.instance)
    assert isinstance(C.game.instance, games.bases.Game)
    C.game.calculate_reward = get_method(hydra_cfg.game.calculate_reward)

    # MCTS namespace
    C.mcts = to_namespace_recurse(to_cont(hydra_cfg.mcts))
    C.mcts.get_node_action = get_method(hydra_cfg.mcts.get_node_action)
    C.mcts.get_node_target_policy = get_method(hydra_cfg.mcts.get_node_target_policy)
    C.mcts.get_node_selection_score = get_method(hydra_cfg.mcts.get_node_selection_score)

    # NETS namespace
    C.nets = to_namespace_recurse(to_cont(hydra_cfg.networks))
    del C.nets.dynamics
    del C.nets.prediction
    del C.nets.representation
    C.nets.factory = SimpleNamespace()
    C.nets.factory.initial_beliefs = functools.partial(
        torch.zeros, tuple(hydra_cfg.networks.beliefs_shape)
    )
    C.nets.factory.initial_latent_rep = functools.partial(
        torch.zeros, tuple(hydra_cfg.networks.latent_rep_shape)
    )
    C.nets.factory.dynamics = pinstantiate(hydra_cfg.networks.dynamics)
    C.nets.factory.prediction = pinstantiate(hydra_cfg.networks.prediction)
    C.nets.factory.representation = pinstantiate(hydra_cfg.networks.representation)
    assert isinstance(C.nets.factory.dynamics(), DynamicsNet)
    assert isinstance(C.nets.factory.prediction(), PredictionNet)
    assert isinstance(C.nets.factory.representation(), RepresentationNet)

    # TRAIN namespace
    C.train = to_namespace_recurse(to_cont(hydra_cfg.training))
    del C.train.optimizer

    def optim_factory(networks: Networks):
        def pgroup(net: NetworkBase, lr: float):
            return {"params": net.parameters(), "lr": C.train.learning_rates.base * lr}

        return instantiate(
            hydra_cfg.training.optimizer,
            [
                pgroup(networks.dynamics, C.train.learning_rates.dynamics),
                pgroup(networks.prediction, C.train.learning_rates.prediction),
                pgroup(networks.representation, C.train.learning_rates.representation),
            ],
        )

    C.train.optimizer_factory = optim_factory

    # PLAYER namespace
    from rl_player import RLPlayer  # this is here to break circular import

    C.player = to_namespace_recurse(to_cont(hydra_cfg.players))
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
