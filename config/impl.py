import os
import abc
import inspect
import logging
import builtins
import functools
from types import UnionType
from typing import Any, Iterable, cast
from collections import defaultdict

import attrs
import hydra
import torch
import omegaconf
from omegaconf import OmegaConf, DictConfig, ListConfig

from utils import optional_map
from games.bases import Game, Player
from config.schema import BaseConfig, NetworkConfig, NetworkSchema
from networks.bases import (
    Networks,
    DynamicsNet,
    NetworkBase,
    PredictionNet,
    RepresentationNet,
)

from . import C

log = logging.getLogger(__name__)


def merge_structured_config_defaults(cfg: Any) -> None:
    """
    This function takes an OmegaConf Config and recursively merges the non-optional
    default values of the underlying structured config classes in-place.
    This is necessary because the user config may override important keys in the schema,
    like _partial_ that control instantiation. Merging the schema defaults ensures the
    correct values of these keys.
    Keys set to None in the schema are Optional and may be overridden by the user, so
    these are not replaced.

    This manual implementation is necessary because Omegaconf disabled auto-expanding of
    nested structured configs, otherwise merging the schema on top of the user config
    would do the trick:
    https://github.com/omry/omegaconf/issues/412
    This manual solution also provides better control over which values are reset.
    """
    if isinstance(cfg, DictConfig):
        for key in cfg:
            if not OmegaConf.is_missing(cfg, key):
                merge_structured_config_defaults(cfg[key])

        t = OmegaConf.get_type(cfg)
        if omegaconf._utils.is_structured_config(t):
            defaults = OmegaConf.structured(t)
            for key in cfg:
                if key in defaults and defaults[key] is not None:
                    OmegaConf.update(cfg, key, defaults[key])  # type: ignore [arg-type]

    elif isinstance(cfg, ListConfig):
        for item in cfg:
            merge_structured_config_defaults(item)


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

    def create_runtime_network_config(net_cfg: NetworkSchema) -> NetworkConfig:
        # empty list [] in config creates None values for belief
        belief_shape = optional_map(tuple)(net_cfg.belief_shape or None)  # type: ignore [arg-type]

        def network_factory() -> Networks:
            return Networks(
                representation=net_cfg.representation(),
                prediction=net_cfg.prediction(),
                dynamics=net_cfg.dynamics(),
                initial_latent=torch.zeros(tuple(net_cfg.latent_shape)),
                initial_belief=optional_map(torch.zeros)(belief_shape),  # type: ignore [arg-type]
            )

        return NetworkConfig(
            factory=network_factory,
            latent_shape=tuple(net_cfg.latent_shape),
            belief_shape=belief_shape,  # type: ignore [arg-type]
        )

    # casts are necessary because here the omegaconf schema types co-exist with the
    # runtime container types - we have to undo some of the type hackery in config.schema
    cfg_obj = cast(BaseConfig, OmegaConf.to_object(cfg))
    C.fill_from(
        attrs.evolve(
            cfg_obj,
            networks=create_runtime_network_config(cast(NetworkSchema, cfg_obj.networks)),
            defaults=None,
        )
    )

    # finally, check for correct class instances:
    # this can only be done after the game instance is in place since network creation may
    # access game-specific data like observation sizes

    assert isinstance(C.game.instance, Game)

    nets = C.networks.factory()

    def net_msg(net_type: str) -> str:
        return f"{net_type} must be subclass of {net_type.upper()}Net!"

    assert isinstance(nets.representation, RepresentationNet), net_msg("representation")
    assert isinstance(nets.prediction, PredictionNet), net_msg("prediction")
    assert isinstance(nets.dynamics, DynamicsNet), net_msg("dynamics")

    def check_players(cls: type | UnionType) -> Iterable[bool]:
        return (
            isinstance(p.func, type) and issubclass(p.func, cls)
            for p in C.players.instances
        )

    # avoid circular imports:
    from rl_player import RLBase

    assert all(check_players(Player | RLBase))
    msg = "There must be at least one RLBase player involved to collect training data!"
    assert any(check_players(RLBase)), msg


def save_source_code() -> None:
    # to yield reproducable experiments, save the source code of all functions and classes
    # and its superclasses referenced in the config

    # {"config namespace": {("config key / origin", "source code")}}
    sources = defaultdict(set)  # type: defaultdict[str, set[tuple[str, str]]]

    def save_recursive(item: Any, path: list[str]) -> None:
        if item is C.training.optimizer or item is C.defaults:
            return

        # partials are probably factories, instantiate them
        # also, C.networks.factory creates the attrs class Networks
        if isinstance(item, functools.partial) or item is C.networks.factory:
            item = item()

        if attrs.has(item):
            for name, child in attrs.asdict(item, recurse=False).items():
                save_recursive(child, path + [name])
            return  # attrs classes are not interesting to save

        namespace = "_".join(path[:-1])

        if inspect.isfunction(item):
            source = inspect.getsource(item)
        elif inspect.isclass(cls := type(item)) and not (
            cls is torch.Tensor or hasattr(builtins, cls.__name__)
        ):
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
                Game,
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