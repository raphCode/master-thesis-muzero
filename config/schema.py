from attrs import define
from omegaconf import MISSING


@define
class Game:
    _target_: str


@define
class GameSchema:
    instance: Game
    calculate_reward: str


@define
class MctsSchema:
    get_node_action: str
    get_node_target_policy: str
    get_node_selection_score: str
    iterations_move_selection: int
    iterations_value_estimate: int


@define
class Net:
    _target_: str


@define
class Tensor:
    size: list[int]
    fill_value: float


@define
class NetworkSchema:
    dynamics: Net
    prediction: Net
    representation: Net
    initial_beliefs: Tensor
    initial_latent_rep: Tensor


@define
class Optimizer:
    _target_: str
    lr: float


@define
class LossWeights:
    latent: float
    absorbing: float
    value: float
    reward: float
    policy: float
    beliefs: float
    player_type: float


@define
class TrainSchema:
    batch_game_size: int
    batch_num_games: int
    discount_factor: float
    n_step_return: int
    replay_buffer_size: int
    max_steps_per_episode: int
    optimizer: Optimizer
    loss_weights: LossWeights


@define
class Player:
    _target_: str


@define
class PlayerSchema:
    instances: list[Player]
    is_teammate: str
