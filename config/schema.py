from typing import TYPE_CHECKING, TypeAlias
from collections.abc import Callable

from attrs import frozen
from omegaconf import MISSING

from networks.bases import Networks

"""
This file creates the config schema, specifying how config items are nested as well as
their names and types.
The dataclasses are also repurposed to function as containers for the global configuration
commonly imported as C, so that static typecheckers can reason about C.
This also means a bit of hackery by switching types based on TYPE_CHECKING.
"""


@frozen
class Instance:
    _target_: str  # fully qualified name of desired class


@frozen
class PartialInstance:
    _target_: str
    _partial_: bool = True


if TYPE_CHECKING:  # RUNTIME TYPES
    # During runtime, the dataclasses are used as containers for the config data.
    # Some of the classes are replaced with instances with actual functionality,
    # so during typechecking use these classes
    from torch.optim import Optimizer

    from games.bases import Game, Player
    from networks.bases import DynamicsNet, PredictionNet, RepresentationNet

    # isort: split
    from fn.action import action_fn
    from fn.policy import policy_fn
    from fn.reward import reward_fn
    from fn.teammate import teammate_fn
    from fn.selection import selection_fn
else:  # OMEGACONF SCHEMA TYPES
    # For omegaconf, just use a dataclass that requires the _target_ config key
    Game = Instance
    Player = PartialInstance
    Optimizer = PartialInstance
    DynamicsNet = PartialInstance
    PredictionNet = PartialInstance
    RepresentationNet = PartialInstance

    # Functions can be python functions or a callable class instances,
    # so actually the type should be str | Instance,
    # but Omegaconf does not support Unions with containers yet.
    # Overriding the str key with a nested config works tho
    fn = str
    action_fn = fn
    policy_fn = fn
    reward_fn = fn
    teammate_fn = fn
    selection_fn = fn


@frozen
class GameConfig:
    instance: Game
    reward_fn: reward_fn


@frozen
class MctsConfig:
    node_action_fn: action_fn
    node_target_policy_fn: policy_fn
    node_selection_score_fn: selection_fn
    iterations_move_selection: int
    iterations_value_estimate: int


@frozen(kw_only=True)
class NetworkConfig:  # runtime config container
    factory: Callable[[], Networks]
    beliefs_shape: tuple[int, ...]
    latent_rep_shape: tuple[int, ...]


@frozen
class NetworkSchema:  # omegaconf schema
    dynamics: DynamicsNet
    prediction: PredictionNet
    representation: RepresentationNet
    beliefs_shape: list[int]
    latent_rep_shape: list[int]


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
    latent: float
    value: float
    reward: float
    policy: float
    player_type: float


@frozen
class TrainConfig:
    batch_game_size: int
    batch_num_games: int
    discount_factor: float
    n_step_return: int
    replay_buffer_size: int
    max_steps_per_episode: int
    optimizer: Optimizer
    learning_rates: LearningRates
    loss_weights: LossWeights


@frozen
class PlayerConfig:
    instances: list[Player]
    is_teammate_fn: teammate_fn


@frozen
class BaseConfig:
    game: GameConfig
    mcts: MctsConfig
    networks: NetworkContainer
    training: TrainConfig
    players: PlayerConfig
    defaults: list[dict[str, str]] = [
        {"game": MISSING},
        {"mcts": MISSING},
        {"networks": MISSING},
        {"training": MISSING},
        {"players": MISSING},
        {"hydra.job": "hydra_job_config"},
    ]
