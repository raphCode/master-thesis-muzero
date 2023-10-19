from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Optional, TypeAlias
from collections.abc import Callable

import attrs
from attrs import define, frozen
from omegaconf import MISSING

"""
This file creates the config schema, specifying how config items are nested as well as
their names and types.
The dataclasses are also repurposed to function as containers for the global configuration
commonly imported as C, so that static typecheckers can reason about C.
This also means a bit of hackery by switching types based on TYPE_CHECKING.
"""

if TYPE_CHECKING:
    from networks import Networks

if not TYPE_CHECKING:
    # with the schema merged first, omegaconf needs the structured config to be writeable:
    # https://github.com/omry/omegaconf/issues/815
    # This hack of redefining frozen is so that the typechecker knows the config is
    # supposed to be immutable
    frozen = define  # noqa: F811


@frozen
class Instance:
    _target_: str  # fully qualified name of desired class


@frozen
class PartialInstance(Instance):
    _partial_: bool = True


@frozen
class PlayerPartialInstance(PartialInstance):
    mcts_cfg: Optional[MctsConfig] = None


if TYPE_CHECKING:  # RUNTIME TYPES
    # During runtime, the dataclasses are used as containers for the config data.
    # Some of the classes are replaced with instances with actual functionality,
    # so during typechecking use the instance or partial types
    from games.bases import Game

    # isort: split
    from torch.optim import Optimizer

    OptimizerPartial = functools.partial[Optimizer]

    from networks.bases import DynamicsNet, PredictionNet, RepresentationNet

    DynamicsNetPartial = functools.partial[DynamicsNet]
    PredictionNetPartial = functools.partial[PredictionNet]
    RepresentationNetPartial = functools.partial[RepresentationNet]

    from fn.action import ActionFn
    from fn.policy import PolicyFn
    from fn.selection import SelectionFn
else:  # OMEGACONF SCHEMA TYPES
    # For omegaconf, just use a dataclass that requires the _target_ config key
    Game = Instance
    OptimizerPartial = PartialInstance

    DynamicsNetPartial = PartialInstance
    PredictionNetPartial = PartialInstance
    RepresentationNetPartial = PartialInstance

    # Functions can be python functions or class instances, so actually the type should be
    # str | Instance
    # but Omegaconf does not support Unions with containers yet:
    fn = Any
    ActionFn = fn
    PolicyFn = fn
    SelectionFn = fn


@frozen
class GameConfig:
    instance: Game


@frozen
class MctsConfig:
    node_action_fn: ActionFn
    node_target_policy_fn: PolicyFn
    node_selection_score_fn: SelectionFn
    iterations: int
    debug_log_mcts_ratio: float


@frozen(kw_only=True)
class NetworkConfig:  # runtime config container
    factory: Callable[[], Networks]
    latent_shape: tuple[int, ...]
    scalar_support_size: int


@frozen
class NetworkSchema:  # omegaconf schema
    dynamics: DynamicsNetPartial
    prediction: PredictionNetPartial
    representation: RepresentationNetPartial
    scalar_support_size: int


if TYPE_CHECKING:  # RUNTIME TYPES
    NetworkContainer: TypeAlias = NetworkConfig
else:  # OMEGACONF SCHEMA TYPES
    NetworkContainer = NetworkSchema


@frozen
class LearningRates:
    base: float
    dynamics: float
    prediction: float
    representation: float


@frozen
class LossWeights:
    value: str
    latent: str
    reward: str
    policy: str
    turn: str


@frozen
class TrainConfig:
    train_selfplay_ratio: float
    batch_size: int
    unroll_length: int
    discount_factor: float
    n_step_horizon: int
    replay_buffer_size: int
    max_steps_per_game: int
    latent_dist_pnorm: float
    optimizer: OptimizerPartial
    learning_rates: LearningRates
    loss_weights: LossWeights


@frozen
class BaseConfig:
    game: GameConfig
    mcts: MctsConfig
    networks: NetworkContainer
    training: TrainConfig
    defaults: Optional[list[Any]] = [
        "_self_",
        {"game": MISSING},
        {"mcts": MISSING},
        {"networks": MISSING},
        {"training": MISSING},
        {"hydra.job": "hydra_job_config"},
    ]

    @classmethod
    def placeholder(cls) -> BaseConfig:
        return cls(
            game=None,  # type:ignore [arg-type]
            mcts=None,  # type:ignore [arg-type]
            networks=None,  # type:ignore [arg-type]
            training=None,  # type:ignore [arg-type]
            defaults=None,
        )

    def fill_from(self, instance: BaseConfig) -> None:
        for field_name in attrs.fields_dict(type(self)):
            object.__setattr__(self, field_name, getattr(instance, field_name))
