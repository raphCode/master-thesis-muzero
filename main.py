from __future__ import annotations

import os
import shutil
import logging
from typing import TYPE_CHECKING

import attrs
import hydra
import torch
from hydra.core.config_store import ConfigStore

from train import Trainer
from config import C
from selfplay import run_episode
from config.impl import (
    populate_config,
    copy_source_code,
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

log = logging.getLogger("main")


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
    copy_source_code()
    if torch.cuda.is_available():
        dev = f"cuda:{torch.cuda.current_device()}"
        log.info(f"Using compute device {dev}")
        torch.set_default_device(dev)

    populate_config(cfg)

    pc = SelfplayPC(C.players.instances)
    rb = ReplayBuffer()
    t = Trainer(pc.net)
    n = 0
    try:
        with TensorboardLogger(log_dir="tb") as tb:

            def unroll_multiline_layout(*tags: str) -> dict[str, tuple[str, list[str]]]:
                return {t: ("Multiline", [t + r"/unroll \d+"]) for t in tags}

            tb.add_custom_scalars_layout(
                unroll={
                    f"loss {k}": (
                        "Multiline",
                        [f"loss: unroll \\d+/{k}"],
                    )
                    for k in attrs.asdict(C.training.loss_weights).keys()
                }
                | unroll_multiline_layout(
                    "latent gradient",
                    "latent cosine similarity",
                    "reward mse",
                    "value mse",
                ),
            )
            pc.net.jit()
            tb.add_graphs(C.networks.factory())
            while True:
                with torch.no_grad():
                    pc.net.eval()
                    result = run_episode(pc, tb.create_step_logger(n))
                n += result.moves
                for traj in result.trajectories:
                    rb.add_trajectory(traj, result.game_completed)
                if True:
                    if rb.fullness == 1:
                        import pickle

                        with open("data.pkl", "wb") as f:
                            pickle.dump(rb, f)
                            return
                    continue
                pc.net.update_rescalers(rb)
                batch_samples = C.training.batch_size * C.training.unroll_length
                target_samples = (
                    rb.data_added * C.training.train_selfplay_ratio * rb.fullness
                )
                pc.net.train()
                while rb.data_sampled < target_samples - batch_samples:
                    t.process_batch(rb.sample(), tb.create_step_logger(n))
    except KeyboardInterrupt:
        if n < 30_000:
            ask_delete_logs()


if __name__ == "__main__":
    monkeypatch_dictconfig()
    register_omegaconf_resolvers()
    main()
