import logging
import functools
from types import UnionType
from typing import Any, Iterable, Optional, cast
from collections.abc import Sequence

import attrs
import hydra
import torch
import gorilla  # type: ignore [import]
import omegaconf
from omegaconf import OmegaConf, DictConfig, ListConfig

from games.bases import Game, Player
from config.schema import BaseConfig, NetworkConfig, NetworkSchema
from networks.bases import (
    Networks,
    DynamicsNetContainer,
    PredictionNetContainer,
    RepresentationNetContainer,
)

from . import C, schema

log = logging.getLogger(__name__)


def assert_callable(obj: Any) -> None:
    """
    Assert that the given object is callable.
    This is intended to check items from the config, so it provides helpful messages it
    the check fails.
    """
    try:
        assert callable(obj)
    except AssertionError as e:
        msg = f"Expected a callable python object, found {type(obj)}:\n{repr(obj)}"
        if isinstance(obj, str):
            msg += "\nIf you wanted to specify the fully qualified name of a function, \
            wrap it in ${fn:...} to resolve it to the python function."
        raise ValueError(msg) from e


def monkeypatch_dictconfig() -> None:
    """
    By default, OmegaConf does not allow merging new keys on top of structured configs.
    This is, OmegaConf.merge(structured, config) will error when 'config' adds new keys.
    This merge order is required in this application to propagate the type information
    from the structured config into the player instances list. Merging the other way
    around is currently buggy:
    https://github.com/omry/omegaconf/issues/1058
    However, new keys can be added when the struct flag is unset on the DictConfig
    instances corresponding to the structured configs. Since the DictConfig instances are
    only created in the merge, the flag can't be set beforehand.
    This therefore monkeypatches DictConfig.__init__() to set the struct flag when a
    structured config of an Instance subclass is to be created. This way the user may add
    new keyword arguments for class instantiation in the config.
    """

    def init_shim(
        self: DictConfig,
        *args: Any,
        flags: Optional[dict[str, bool]] = None,
        ref_type: Any = Any,
        **kwargs: Any,
    ) -> Any:
        if isinstance(ref_type, type) and issubclass(ref_type, schema.Instance):
            flags = flags or {}
            flags["struct"] = False
        return original_init(self, *args, flags=flags, ref_type=ref_type, **kwargs)

    gorilla.apply(
        gorilla.Patch(DictConfig, "__init__", init_shim, gorilla.Settings(allow_hit=True))
    )
    original_init = gorilla.get_original_attribute(DictConfig, "__init__")


def register_omegaconf_resolvers() -> None:
    # This resolver allows accessing python functions by their fully qualified name:
    OmegaConf.register_new_resolver("fn", hydra.utils.get_method)


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

    assert_callable(cfg.game.reward_fn)
    assert_callable(cfg.mcts.node_action_fn)
    assert_callable(cfg.mcts.node_target_policy_fn)
    assert_callable(cfg.mcts.node_selection_score_fn)

    def create_runtime_network_config(net_cfg: NetworkSchema) -> NetworkConfig:
        initial_tensor = functools.partial(torch.rand, requires_grad=True)

        def shape_check(shape: Sequence[int], name: str) -> None:
            assert len(shape) > 0, f"{name} shape does not contain dimensions!"
            assert all(
                dim >= 0 for dim in shape
            ), f"{name} shape contains negative dimensions!"

        latent_shape = tuple(net_cfg.latent_shape)
        belief_shape = tuple(net_cfg.belief_shape)
        shape_check(latent_shape, "latent")
        shape_check(belief_shape, "belief")

        def network_factory() -> Networks:
            return Networks(
                representation=RepresentationNetContainer(net_cfg.representation()),
                prediction=PredictionNetContainer(net_cfg.prediction()),
                dynamics=DynamicsNetContainer(net_cfg.dynamics()),
                initial_latent=initial_tensor(latent_shape),
                initial_belief=initial_tensor(belief_shape),
            )

        return NetworkConfig(
            factory=network_factory,
            latent_shape=latent_shape,
            belief_shape=belief_shape,
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

    # Network container check the implementation types themselves
    C.networks.factory()

    def net_msg(net_type: str) -> str:
        return f"{net_type} must be subclass of {net_type.upper()}Net!"

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
