import operator

import attrs
import numpy as np
import torch

from util import RingBuffer, TensorCache
from config import C
from trajectory import TrainingData, TrajectoryState

rng = np.random.default_rng()


class ReplayBuffer:
    """
    Stores the latest trajectory states in a buffer and samples training batches from it.
    The trajectory states are stored in a contiguous buffer, together with an unique id to
    detect when a trajectory ends and a new one starts.
    """

    def __init__(self) -> None:
        # the integer is a unique id to differentiate different trajectories
        self.buffer = RingBuffer[tuple[int, TrainingData]](C.training.replay_buffer_size)
        self.cache = TensorCache()
        self.discounts = np.concatenate(
            (
                [1],
                np.cumprod(np.full(C.training.n_step_return, C.training.discount_factor)),
            )
        )

    def add_trajectory(self, traj: list[TrajectoryState], game_completed: bool) -> None:
        # The unique trajectory id could also be an increasing number, but I don't like a
        # counter growing indefinitely.
        # An id is derived from the start index in the buffer and the trajectory length.
        # Effectively this calculates the end index in the buffer (actually the index
        # after the trajectory end, and without modulo len(buffer)).
        # In a ring buffer all entries leading up to the end index are written to, a
        # trajectory with the same id as an existing one overwrites the old one fully,
        # ensuring the id is indeed unique.
        traj_id = self.buffer.position + len(traj)
        next_rewards = np.array([ts.reward for ts in traj][1:])
        for n, ts in enumerate(traj):
            nstep_idx = min(len(traj) - 1, n + C.training.n_step_return)
            if nstep_idx == len(traj) - 1 and game_completed:
                value_target = 0
            else:
                value_target = traj[nstep_idx].mcts_value * self.discounts[nstep_idx - n]

            value_target += np.inner(
                next_rewards[n:nstep_idx], self.discounts[1 : nstep_idx - n + 1]
            )
            self.buffer.append(
                (
                    traj_id,
                    TrainingData.from_trajectory_state(
                        ts, value_target, is_initial=(n == 0), cache=self.cache
                    ),
                )
            )

    def sample(self) -> list[list[TrajectoryState]]:
        lens = np.array(self.lens)
        probs = lens / lens.sum()
        batch_trajs = []
        data = np.empty(len(self.data), dtype=object)
        data[:] = self.data
        for traj in rng.choice(data, size=C.training.batch_size, p=probs):
            i = rng.integers(len(traj))
            batch_trajs.append(
                (traj + self.empty_batch_game)[i : i + C.training.trajectory_length]
            )

        # transpose: outer dim: batch_size -> trajectory_length
        batch_steps = zip(*batch_trajs)
        field_names = tuple(map(operator.attrgetter("name"), attrs.fields(TrainingData)))

        batch_train_data = []
        for steps in batch_steps:
            # unpack TrainingData classes into tuples
            unpacked_steps = map(attrs.astuple, steps)
            # transpose: outer dim: batch_size -> len(field_names)
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

    def __len__(self) -> int:
        return len(self.buffer)
