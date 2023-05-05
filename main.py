import logging
import contextlib

import hydra
import torch
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
from torch.utils.tensorboard import SummaryWriter  # type: ignore [attr-defined]

from train import Trainer
from config import C
from selfplay import run_episode
from config.impl import (
    populate_config,
    save_source_code,
    monkeypatch_dictconfig,
    register_omegaconf_resolvers,
)
from config.schema import BaseConfig
from replay_buffer import ReplayBuffer
from player_controller import SinglePC

cs = ConfigStore.instance()
cs.store(name="hydra_job_config", group="hydra.job", node={"chdir": True})
cs.store(name="base_config", node=BaseConfig)

log = logging.getLogger("main")


@hydra.main(version_base=None, config_path="run_config", config_name="base_config")
def main(cfg: DictConfig) -> None:
    logging.captureWarnings(True)
    populate_config(cfg)
    save_source_code()

    pc = SinglePC(C.players.instances)
    rb = ReplayBuffer()
    t = Trainer(pc.net)
    n = 0
    with contextlib.closing(SummaryWriter(log_dir="tb")) as sw:  # type: ignore [no-untyped-call] # noqa: E501
        while n < 100_000:
            with torch.no_grad():
                result = run_episode(pc)
            log.info(f"Finished selfplay game ({result.moves} steps)")
            sw.add_scalar("selfplay/game length", result.moves, n)  # type: ignore [no-untyped-call] # noqa: E501
            sw.add_scalar("selfplay/reward", result.trajectories[0][-1].reward, n)  # type: ignore [no-untyped-call] # noqa: E501
            n += result.moves
            for traj in result.trajectories:
                rb.add_trajectory(traj, result.game_completed)
            if len(rb) > 0.2 * C.training.replay_buffer_size:
                t.process_batch(rb.sample(), sw, n)


if __name__ == "__main__":
    monkeypatch_dictconfig()
    register_omegaconf_resolvers()
    main()
