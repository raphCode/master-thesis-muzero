from attrs import frozen
from omegaconf import MISSING


@frozen
class Game:
    _target_: str


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


@frozen
class Net:
    _target_: str


@frozen
class NetworkSchema:
    dynamics: Net
    prediction: Net
    representation: Net
    beliefs_shape: list[int]
    latent_rep_shape: list[int]


@frozen
class Optimizer:
    _target_: str


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
class Player:
    _target_: str


@frozen
class PlayerConfig:
    instances: list[Player]
    is_teammate_fn: str


@frozen
class BaseConfig:
    game: GameConfig
    mcts: MctsConfig
    networks: NetworkSchema
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
