from attrs import define


@define
class Game:
    _target_: str


@define
class GameSchema:
    instance: Game
    reward_fn: str


@define
class MctsSchema:
    node_action_fn: str
    node_target_policy_fn: str
    node_selection_score_fn: str
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
    beliefs_shape: list[int]
    latent_rep_shape: list[int]


@define
class Optimizer:
    _target_: str


@define
class LearningRates:
    base: float
    dynamics: float
    prediction: float
    representation: float


@define
class LossWeights:
    latent: float
    value: float
    reward: float
    policy: float
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
    learning_rates: LearningRates
    loss_weights: LossWeights


@define
class Player:
    _target_: str


@define
class PlayerSchema:
    instances: list[Player]
    is_teammate_fn: str
