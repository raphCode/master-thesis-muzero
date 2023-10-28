import random
import logging

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

    # Monotonous raising counters representing the number of incoming and outgoing samples
    data_added: int
    data_sampled: int

    def __init__(self) -> None:
        # the integer is a unique id to differentiate different trajectories
        self.buffer = RingBuffer[tuple[int, TrainingData]](C.training.replay_buffer_size)
        scalar_buffer_size = (
            C.training.replay_buffer_size * C.game.instance.max_num_players
        )
        self.values = RingBuffer[float](scalar_buffer_size)
        self.rewards = RingBuffer[float](scalar_buffer_size)
        self.cache = TensorCache()
        self.data_added = 0
        self.data_sampled = 0
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

        rewards = np.stack([ts.reward for ts in traj])
        for t, traj_state in enumerate(traj):
            # The target for the prediction network value head is the bootstrapped n step
            # return. In general, it is calculated by adding future rewards over the n
            # step horizon and a final value estimate.
            #
            # In the MuZero paper, this notation is used:
            # In state s_t the action a_(t+1) was taken, yielding reward r_(t+1).
            # The MCTS in s_t produced a value of v_t.
            # The value target z_t with horizon size n is calculated by:
            # z_t = d^0 * r_(t+1) + d^1 * r_(t+2), ..., d^(n-1) * r_(t+n) + d^n * v_(t+n)
            #
            # In this implementation the trajectory list at index stores at index t:
            # - state s_t
            # - action a_(t+1)
            # - reward r_(t+1)
            # - mcts root node value v_t
            #
            # Therefore, in the implementation, 1 is subtracted from the reward indices
            # compared to the formula above.
            # The discount factor d raised to the power x is stored at self.discounts[x]
            remaining_len = len(traj) - t
            n = min(C.training.n_step_horizon, remaining_len)

            value_target = np.tensordot(
                rewards[t : t + n], self.discounts[:n], axes=(0, 0)
            )

            nstep_end_terminal = game_completed and t + n == len(traj)
            if not nstep_end_terminal:
                value_target += traj[t + n - 1].mcts_value * self.discounts[n]

            self.buffer.append(
                (
                    traj_id,
                    TrainingData.from_trajectory_state(
                        traj_state,
                        value_target,
                        is_terminal=(game_completed and t == len(traj) - 1),
                        cache=self.cache,
                    ),
                )
            )
            self.values.extend(value_target)
            self.rewards.extend(traj_state.reward)
        self.data_added += len(traj)

    def sample(self) -> list[TrainingData]:
        """
        Return a training batch composed of random subsections of the stored trajectories.
        See the docs for TrainingData for an ascii art visualisation of the returned data.
        """
        unroll_len = random.randrange(1, C.training.unroll_length + 1)
        # Select random start indices in the buffer for the trajectory subsections.
        # A subsection is only accepted when when the remaining trajectory has the the
        # required minimum length.
        # This makes it necessary to retry the sampling until the batch size is filled.
        retry_limit = 10
        sampled_starts = set()
        for _ in range(retry_limit * C.training.batch_size):
            start_index = random.randrange(len(self.buffer))
            traj_id, traj = self.buffer[start_index]
            unroll_len_id, _ = self.buffer[start_index + unroll_len - 1]
            if traj.is_observation and traj_id == unroll_len_id:
                sampled_starts.add(start_index)
            if len(sampled_starts) == C.training.batch_size:
                break
        else:
            log.warning(
                f"Underfull batch size: {len(sampled_starts)}/{C.training.batch_size}"
            )

        # Walk forward in the buffer from the start indices, collecting trajectory states
        batch = []
        for n in range(unroll_len):
            batch_step = []
            for start_index in sorted(sampled_starts):
                _, data = self.buffer[start_index + n]
                batch_step.append(data)
            batch.append(TrainingData.stack_batch(batch_step))
        self.data_sampled += len(sampled_starts) * C.training.unroll_length
        return batch

    @property
    def reward_bounds(self) -> tuple[float, float]:
        """
        Minimum and maximum rewards found in the stored data.
        """
        return min(self.rewards), max(self.rewards)

    @property
    def value_bounds(self) -> tuple[float, float]:
        """
        Minimum and maximum value targets found in the stored data.
        """
        return min(self.values), max(self.values)

    @property
    def fullness(self) -> float:
        """
        Number from zero to one describing how full the buffer is.
        """
        return len(self.buffer) / self.buffer.size

    def __len__(self) -> int:
        return len(self.buffer)
