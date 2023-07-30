from __future__ import annotations

import os
import shutil
import logging
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore

from train import Trainer
from config import C
from selfplay import run_episode
from config.impl import (
    populate_config,
    monkeypatch_dictconfig,
    register_omegaconf_resolvers,
)
from config.schema import BaseConfig
from replay_buffer import ReplayBuffer
from player_controller import SelfplayPC
from tensorboard_wrapper import TensorboardLogger

if TYPE_CHECKING:
    from omegaconf import DictConfig

cs = ConfigStore.instance()
cs.store(name="hydra_job_config", group="hydra.job", node={"chdir": True})
cs.store(name="base_config", node=BaseConfig)


@hydra.main(version_base=None, config_path="run_config", config_name="base_config")
def main(cfg: DictConfig) -> None:
    def ask_delete_logs() -> None:
        cwd = os.getcwd()
        orig_cwd = hydra.utils.get_original_cwd()
        assert cwd.startswith(orig_cwd) and len(cwd) > len(orig_cwd)
        assert "outputs" in cwd
        assert os.path.exists("tb")
        assert os.path.exists(".hydra")
        msg = (
            "\nAborted a short run."
            f"\nThe run created tensorboard event files and various logs in:\n{cwd}"
            "\nDo you want to delete this directory? [N/y] "
        )
        if input(msg) == "y":
            shutil.rmtree(cwd)

    logging.captureWarnings(True)
    populate_config(cfg)

    torch.autograd.set_detect_anomaly(True)  # type: ignore [attr-defined]
    torch.set_default_dtype(torch.float64)  # type: ignore [no-untyped-call]
    np.set_printoptions(precision=3, suppress=True)

    log_dir = "tb"
    if "tb_name" in cfg:
        log_dir = cfg["tb_name"]
    if cfg.no_tb:
        log_dir = "/tmp"

    pc = SelfplayPC(C.players.instances)
    rb = ReplayBuffer()
    t = Trainer(pc.net)

    n = 0
    try:
        with TensorboardLogger(log_dir="tb") as tb:
            # tb.add_graphs(pc.net)

            pc.net.jit()
            tb.add_graphs(C.networks.factory())
            while n < 1_000_000:
                with torch.no_grad():
                    result = run_episode(pc, tb.create_step_logger(n))
                n += result.moves
                for traj in result.trajectories:
                    rb.add_trajectory(traj, result.game_completed)
                pc.net.update_rescalers(rb)

                batch_samples = C.training.batch_size * C.training.max_trajectory_length
                target_samples = (
                    rb.data_added * C.training.train_selfplay_ratio * rb.fullness
                )
                while rb.data_sampled < target_samples - batch_samples:
                    t.process_batch(rb.sample(), tb.create_step_logger(n))
    except KeyboardInterrupt:
        if n < 30_000:
            ask_delete_logs()
    # print("selfplay:", timer_selfplay.summary())
    # print("train:", t.timer.summary())


if __name__ == "__main__":
    monkeypatch_dictconfig()
    register_omegaconf_resolvers()
    main()
    print(os.getcwd())
    pass
