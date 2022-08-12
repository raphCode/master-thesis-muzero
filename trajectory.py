import functools
import operator
from enum import IntEnum
from typing import Deque, Optional
from collections import deque
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from attrs import  frozen
import attrs

from config import config as C


class PlayerType(IntEnum):
    """
    Reprents whose turn it is as seen by a specific player.
    This is important in the recorded trajectories and in the tree search.
    The chance player models randomness in a game by taking a random action.
    """

    Self = 0
    Chance = 1
    Opponent = 2
    Teammate = 3


@frozen(kw_only=True)
class TrajectoryState:
    observation: Optional[tuple[torch.Tensor]]
    latent_rep: Optional[torch.Tensor]
    old_beliefs: Optional[torch.Tensor]  # prior to the representation inference
    dyn_beliefs: Optional[torch.Tensor]  # after the dynamics inference
    player_type: PlayerType
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
    beliefs: torch.Tensor  # either dyn_beliefs or old_beliefs, depending on observation
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
            ([1], np.cumprod(np.full(C.train.n_step_return - 1, C.train.discount_factor)))
        )
        self.empty_observation = tuple(
            torch.zeros(s) for s in C.game.instance.observation_shapes
        )
        self.empty_latent_rep = torch.zeros_like(C.nets.initial_latent_rep)
        self.empty_batch_game = [
            TrainingData(
                is_observation=torch.tensor(False),
                is_data=torch.tensor(False),
                observation=self.empty_observation,
                latent_rep=self.empty_latent_rep,
                beliefs=torch.zeros_like(C.nets.initial_beliefs),
                player_type=torch.tensor(0),
                action_onehot=torch.zeros(C.game.instance.max_num_actions),
                target_policy=torch.zeros(C.game.instance.max_num_actions),
                value_target=torch.tensor(0),
                reward=torch.tensor(0),
            )
        ] * C.train.batch_game_size

    def add_trajectory(self, traj: list[TrajectoryState], game_terminated: bool):
        int64t = functools.partial(torch.tensor, dtype=torch.int64)
        floatt = functools.partial(torch.tensor, dtype=torch.float)

        rewards = np.array([ts.reward for ts in traj])

        train_data = []
        for n, ts in enumerate(traj):
            nstep_idx = min(len(traj), n + C.train.n_step_return)
            if nstep_idx == len(traj) and game_terminated:
                value_target = 0
            else:
                nstep_idx -= 1
                value_target = traj[nstep_idx].value * self.discounts[nstep_idx - n]

            value_target += np.inner(
                rewards[n:nstep_idx], self.discounts[: nstep_idx - n]
            )
            is_obs = ts.observation is not None
            train_data.append(
                TrainingData(
                    is_observation=torch.tensor(is_obs),
                    is_data=torch.tensor(True),
                    observation=ts.observation if is_obs else self.empty_observation,
                    latent_rep=ts.latent_rep if not is_obs else self.empty_latent_rep,
                    beliefs=ts.old_beliefs if is_obs else ts.dyn_beliefs,
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

        batch_train_data=[]
        for steps in batch_steps:
            # unpack TrainingData classes into tuples
            unpacked_steps = map(attrs.astuple, steps)
            # transpose: outer dim: batch_num_games -> len(field_names)
            batch_fields = zip(*unpacked_steps)
            fields=dict()
            for name, batch in zip(field_names, batch_fields):
                if name == "observation":
                    data=tuple(map(torch.stack, zip(*batch)))
                elif name == "player_type":
                    data=torch.stack(batch)
                else:
                    data=torch.vstack(batch)
                fields[name]=data
            batch_train_data.append(TrainingData(**fields))
                    


    def __len__(self):
        return len(self.data)
