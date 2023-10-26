from __future__ import annotations

import os
import shutil
import logging
import traceback
from typing import TYPE_CHECKING
from contextlib import suppress

import attrs
import hydra
import torch
from hydra.core.config_store import ConfigStore

from train import Trainer
from config import C
from selfplay import run_episode,random_play
from config.impl import (
    populate_config,
    copy_source_code,
    patch_dictconfig,
    monkeypatch_dictconfig,
    register_omegaconf_resolvers,
)
from config.schema import BaseConfig
from replay_buffer import ReplayBuffer
from tensorboard_wrapper import TensorboardLogger

if TYPE_CHECKING:
    from omegaconf import DictConfig

cs = ConfigStore.instance()
cs.store(name="hydra_job_config", group="hydra.job", node={"chdir": True})
cs.store(name="base_config", node=BaseConfig)

log = logging.getLogger("main")


# @patch_dictconfig
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

    def set_terminal_title() -> None:
        relpath = os.path.relpath(os.curdir, start=hydra.utils.get_original_cwd())
        cwd = os.getcwd()
        title = min(cwd, relpath, key=len)
        print(f"\x1b]2;{title}\x07")

    set_terminal_title()

    logging.captureWarnings(True)
    copy_source_code()
    if torch.cuda.is_available():
        dev = f"cuda:{torch.cuda.current_device()}"
        log.info(f"Using compute device {dev}")
        torch.set_default_device(dev)  # type: ignore [no-untyped-call]

    populate_config(cfg)

    nets_selfplay = C.networks.factory()
    nets = C.networks.factory()
    nets_selfplay.eval()
    nets.train()
    if torch.cuda.is_available():
        nets.cuda()

    for ps, p in zip(nets_selfplay.parameters(), nets.parameters()):
        ps.data = p.data  # share underlying parameter tensors
    # trace in eval and in training mode
    nets_selfplay.jit()
    nets.jit()

    rb = ReplayBuffer()
    t = Trainer(nets)
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
                | unroll_multiline_layout(  # type: ignore [operator]
                    "latent gradient",
                    "latent cosine similarity",
                ),
            )
            nets.jit()
            tb.add_graphs(C.networks.factory())
            while True:
                if n < C.training.random_play_steps:
                    result = random_play(tb.create_step_logger(n))
                else:
                    with torch.no_grad():
                        result = run_episode(nets_selfplay, tb.create_step_logger(n))
                n += result.moves
                rb.add_trajectory(result.trajectory, result.game_completed)
                nets.update_rescalers(rb)
                batch_samples = C.training.batch_size * C.training.unroll_length
                target_samples = (
                    rb.data_added * C.training.train_selfplay_ratio * rb.fullness
                )
                while rb.data_sampled < target_samples - batch_samples:
                    t.process_batch(rb.sample(), tb.create_step_logger(n))
    except (KeyboardInterrupt, Exception) as e:
        log.error(repr(e) + "\n" + traceback.format_exc())
        if n < 300_000:
            with suppress(AssertionError):
                ask_delete_logs()
        raise


if __name__ == "__main__":
    monkeypatch_dictconfig()
    register_omegaconf_resolvers()
    main()
