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
else:  # OMEGACONF SCHEMA TYPES
    # For omegaconf, just use a dataclass that requires the _target_ config key
    Game = Instance
    Player = PartialInstance
    Optimizer = PartialInstance
    DynamicsNet = PartialInstance
    PredictionNet = PartialInstance
    RepresentationNet = PartialInstance


@frozen
class GameConfig:
    instance: Game
    reward_fn: str


@frozen
class MctsConfig:
    node_action_fn: str
    node_target_policy_fn: str
    node_selection_score_fn: str
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
    is_teammate_fn: str


@frozen
class BaseConfig:
    game: GameConfig
    mcts: MctsConfig
    networks: NetworkContainer
    training: TrainConfig
    players: PlayerConfig
    defaults: list = [
        {"game": MISSING},
        {"mcts": MISSING},
        {"networks": MISSING},
        {"training": MISSING},
        {"players": MISSING},
        {"hydra.job": "hydra_job_config"},
    ]
