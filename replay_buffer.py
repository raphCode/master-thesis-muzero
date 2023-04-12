import random
import logging
from typing import cast

import numpy as np

from util import RingBuffer, TensorCache
from config import C
from trajectory import TrainingData, TrajectoryState

log = logging.getLogger(__name__)


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
        # These are pre-raised discount factors:
        # discounts[x] = pow(discount_factor, x)
        self.discounts = np.concatenate(
            (
                [1],
                np.cumprod(
                    np.full(C.training.n_step_horizon, C.training.discount_factor)
                ),
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

        rewards = np.array([ts.reward for ts in traj])
        for n, ts in enumerate(traj):
            # The target for the prediction network value head is the bootstrapped n step
            # return. It is calculated by adding future rewards over the n step horizon
            # and a final value estimate:
            # - sum up the next n rewards:
            #   beginning with the current state s_0, up to s_(n-1)
            # - add the value estimate of the ensuing state s_n:
            #   - terminal states have a value estimate of zero and can be omitted,
            #     freeing up the last state to be included in the reward sum.
            #   - if the game was aborted prematurely, the last state is not terminal:
            #     we need to use its value estimate, so we may only sum up the rewards up
            #     to the second to last state
            remaining_nstep_len = len(traj) - n - (not game_completed)
            nstep_len = min(C.training.n_step_horizon, remaining_nstep_len)

            # All summands are discounted with the discount factor, raised to how many
            # steps in the future they appear.
            value_target = np.inner(
                rewards[n : n + nstep_len], self.discounts[:nstep_len]
            )
            # The nstep horizon length is calculated carefully: When the game completed
            # normally, the horizon end may lie outside the trajectory, ensuring the value
            # estimate is not added on terminal states:
            if n + nstep_len < len(traj):
                value_target += traj[n + nstep_len].mcts_value * self.discounts[nstep_len]

            self.buffer.append(
                (
                    traj_id,
                    TrainingData.from_trajectory_state(
                        ts, value_target, is_initial=(n == 0), cache=self.cache
                    ),
                )
            )

    def sample(self) -> list[TrainingData]:
        """
        Return a training batch composed of random subsections of the stored trajectories.
        See the docs for TrainingData for an ascii art visualisation of the returned data.
        """
        # Select random start indices in the buffer for the trajectory subsections.
        # A subsection is only accepted when when the remaining trajectory has the the
        # required minimum length.
        # This makes it necessary to retry the sampling until the batch size is filled.
        retry_limit = 10
        sampled_starts = set()
        for _ in range(retry_limit * C.training.batch_size):
            start_index = random.randrange(len(self.buffer))
            traj_id, _ = self.buffer[start_index]
            minlen_id, _ = self.buffer[start_index + C.training.min_trajectory_length - 1]
            if traj_id == minlen_id:
                sampled_starts.add((start_index, traj_id))
            if len(sampled_starts) == C.training.batch_size:
                break
        else:
            log.warning(
                f"Underfull batch size: {len(sampled_starts)}/{C.training.batch_size}"
            )

        # Walk forward in the buffer from the start indices, collecting the trajectory
        # states until the maximum length is reached or all trajectories ended.
        batch = []
        for n in range(C.training.max_trajectory_length):
            batch_step = []
            is_data = False
            for start_index, start_traj_id in sorted(sampled_starts):
                traj_id, data = self.buffer[start_index + n]
                if traj_id == start_traj_id:
                    is_data = True
                else:
                    data = cast(TrainingData, TrainingData.dummy)
                batch_step.append(data)
            if not is_data:
                # all trajectories ended early
                break
            batch.append(TrainingData.stack_batch(batch_step))
        return batch

    def __len__(self) -> int:
        return len(self.buffer)
