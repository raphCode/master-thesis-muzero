import os
import logging
import itertools
import contextlib
from typing import Any

import hydra
import torch
from attrs import define
from omegaconf import MISSING, OmegaConf, DictConfig
from hydra.core.config_store import ConfigStore
from torch.utils.tensorboard import SummaryWriter

import config
import selfplay
from train import process_batch
from config import config as C
from trajectory import ReplayBuffer
from config.schema import GameSchema, MctsSchema, TrainSchema, PlayerSchema, NetworkSchema

cs = ConfigStore.instance()
cs.store(name="hydra_job_config", group="hydra.job", node={"chdir": True})


@define
class BaseConfig:
    game: GameSchema
    mcts: MctsSchema
    networks: NetworkSchema
    training: TrainSchema
    players: PlayerSchema
    defaults: list[Any] = [
        {"game": MISSING},
        {"mcts": MISSING},
        {"networks": MISSING},
        {"training": MISSING},
        {"players": MISSING},
        {"hydra.job": "hydra_job_config"},
    ]


cs.store(name="base_config", node=BaseConfig)


log = logging.getLogger("main")


@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(cfg: DictConfig):
    config.populate_config(cfg)
    config.save_source_code()

    if "load_checkpoint" in cfg:
        c = torch.load(hydra.utils.to_absolute_path(cfg.load_checkpoint))
        C.nets.representation.load_state_dict(c["nets"]["representation"])
        C.nets.prediction.load_state_dict(c["nets"]["prediction"])
        C.nets.dynamics.load_state_dict(c["nets"]["dynamics"])
        C.train.optimizer.load_state_dict(c["optimizer"])

    rb = ReplayBuffer()
    os.mkdir("checkpoints")

    log_dir = "tb"
    C.tb = True
    if "no_tb" in cfg:
        C.tb = False
        log_dir = "/tmp"

    with contextlib.closing(SummaryWriter(log_dir=log_dir)) as sw:
        # TODO: try inference mode to speed up things
        for n in itertools.count(0):
            if n % 10 == 0:
                with torch.no_grad():
                    selfplay.run_episode(rb, sw, n)
            if len(rb) > 0.1 * C.train.replay_buffer_size:
                loss = process_batch(rb.sample(), sw, n)
                log.info(f"Finished batch update (loss: {loss.item():.5f})")
            if n % 100 == 0:
                torch.save(
                    {
                        "nets": {
                            "representation": C.nets.representation.state_dict(),
                            "prediction": C.nets.prediction.state_dict(),
                            "dynamics": C.nets.dynamics.state_dict(),
                        },
                        "optimizer": C.train.optimizer.state_dict(),
                    },
                    f"checkpoints/{n//100:06d}.pt",
                )
                log.info(f"Saved checkpoint!")


if __name__ == "__main__":
    main()
