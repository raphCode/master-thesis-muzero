from __future__ import annotations

import os
import pickle
import shutil
import logging
from typing import TYPE_CHECKING
from itertools import count

import attrs
import hydra
import torch
from hydra.core.config_store import ConfigStore

from train import Trainer
from config import C
from config.impl import (
    populate_config,
    copy_source_code,
    monkeypatch_dictconfig,
    register_omegaconf_resolvers,
)
from config.schema import BaseConfig
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

    log.info("Loading data...")
    with open("/home/raph/master_thesis/implementation/data.pkl", "rb") as f:
        rb = pickle.load(f)
    net = C.networks.factory()
    t = Trainer(net)
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
            # net.jit()
            tb.add_graphs(C.networks.factory())
            net.update_rescalers(rb)
            net.train()
            for n in count(1):
                t.process_batch(rb.sample(), tb.create_step_logger(n))
    except KeyboardInterrupt:
        if n < 30_000:
            ask_delete_logs()


if __name__ == "__main__":
    monkeypatch_dictconfig()
    register_omegaconf_resolvers()
    main()
