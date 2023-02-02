import operator
import functools
from typing import Deque, Optional, TypeAlias
from collections import deque
from collections.abc import Sequence

import attrs
import numpy as np
import torch
import torch.nn.functional as F
from attrs import frozen
from torch import Tensor

from config import C


@frozen
class InitialTensor:
    """
    Indicates that the initial belief or latent should be used.
    This must be encoded in a special type because information may get sent between
    multiple processes, so sentinel values in form of object instances won't necessarily
    compare equals in different processes.
    """

    pass


@frozen
class ObservationInfo:
    observations: tuple[Tensor, ...]
    belief: Optional[Tensor | InitialTensor]


@frozen
class LatentInfo:
    latent: Tensor | InitialTensor
    belief: Optional[Tensor | InitialTensor]


InfoType: TypeAlias = ObservationInfo | LatentInfo


@frozen(kw_only=True)
class TrajectoryState:
    info: InfoType
    current_player: int
    action: int
    target_policy: Sequence[float]
    mcts_value: float
    reward: float


@frozen(kw_only=True)
class TrainingData:
    is_observation: torch.Tensor
    is_data: torch.Tensor
    observation: tuple[torch.Tensor]
    latent_rep: torch.Tensor
    beliefs: torch.Tensor  # current or previous beliefs, depending on is_observation
    player_type: torch.Tensor
    action_onehot: torch.Tensor
    target_policy: torch.Tensor
    value_target: torch.Tensor
    reward: torch.Tensor


rng = np.random.default_rng()


class ReplayBuffer:
    lens = Deque[int]
    data = Deque[list[TrajectoryState]]

    def __init__(self):
        self.lens = deque(maxlen=C.train.replay_buffer_size)
        self.data = deque(maxlen=C.train.replay_buffer_size)
        self.discounts = np.concatenate(
            ([1], np.cumprod(np.full(C.train.n_step_return, C.train.discount_factor)))
        )
        self.empty_observation = tuple(
            torch.zeros(s, dtype=torch.float) for s in C.game.instance.observation_shapes
        )
        self.empty_latent_rep = torch.zeros(C.nets.latent_rep_shape)
        self.empty_batch_game = [
            TrainingData(
                is_observation=torch.tensor(False),
                is_data=torch.tensor(False),
                observation=self.empty_observation,
                latent_rep=self.empty_latent_rep,
                beliefs=torch.zeros(C.nets.beliefs_shape),
                player_type=torch.tensor(0),
                action_onehot=torch.zeros(C.game.instance.max_num_actions),
                target_policy=torch.zeros(C.game.instance.max_num_actions),
                value_target=torch.tensor(0.0),
                reward=torch.tensor(0.0),
            )
        ] * C.train.batch_game_size

    def add_trajectory(self, traj: list[TrajectoryState], game_terminated: bool):
        int64t = functools.partial(torch.tensor, dtype=torch.int64)
        floatt = functools.partial(torch.tensor, dtype=torch.float)

        next_rewards = np.array([ts.reward for ts in traj][1:])

        train_data = []
        for n, ts in enumerate(traj):
            nstep_idx = min(len(traj) - 1, n + C.train.n_step_return)
            if nstep_idx == len(traj) - 1 and game_terminated:
                value_target = 0
            else:
                value_target = traj[nstep_idx].mcts_value * self.discounts[nstep_idx - n]

            value_target += np.inner(
                next_rewards[n:nstep_idx], self.discounts[1 : nstep_idx - n + 1]
            )
            is_obs = isinstance(ts.info, ObservationInfo)
            train_data.append(
                TrainingData(
                    is_observation=torch.tensor(is_obs),
                    is_data=torch.tensor(True),
                    observation=ts.info.observation if is_obs else self.empty_observation,
                    latent_rep=self.empty_latent_rep if is_obs else ts.info.latent_rep,
                    beliefs=ts.info.prev_beliefs if is_obs else ts.info.beliefs,
                    player_type=int64t(ts.player_type),
                    action_onehot=F.one_hot(
                        int64t(ts.action), C.game.instance.max_num_actions
                    ),
                    target_policy=floatt(ts.target_policy),
                    value_target=floatt(value_target),
                    reward=floatt(ts.reward),
                )
            )

        self.lens.append(len(train_data))
        self.data.append(train_data)

    def sample(self) -> list[list[TrajectoryState]]:
        lens = np.array(self.lens)
        probs = lens / lens.sum()
        batch_trajs = []
        data = np.empty(len(self.data), dtype=object)
        data[:] = self.data
        for traj in rng.choice(data, size=C.train.batch_num_games, p=probs):
            i = rng.integers(len(traj))
            batch_trajs.append(
                (traj + self.empty_batch_game)[i : i + C.train.batch_game_size]
            )

        # transpose: outer dim: batch_num_games -> batch_game_size
        batch_steps = zip(*batch_trajs)
        field_names = tuple(map(operator.attrgetter("name"), attrs.fields(TrainingData)))

        batch_train_data = []
        for steps in batch_steps:
            # unpack TrainingData classes into tuples
            unpacked_steps = map(attrs.astuple, steps)
            # transpose: outer dim: batch_num_games -> len(field_names)
            batch_fields = zip(*unpacked_steps)
            fields = dict()
            for name, batch in zip(field_names, batch_fields):
                # TODO: save memory by setting latent_rep,beliefs = None for all steps expect first
                if name == "observation":
                    data = tuple(map(torch.stack, zip(*batch)))
                elif name in ("is_observation", "is_data", "player_type"):
                    data = torch.stack(batch)
                else:
                    data = torch.vstack(batch)
                fields[name] = data
            batch_train_data.append(TrainingData(**fields))

        return batch_train_data

    def __len__(self):
        return len(self.data)
