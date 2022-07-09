from typing import Any

import hydra
from attrs import define
from omegaconf import MISSING, OmegaConf, DictConfig
from hydra.core.config_store import ConfigStore

import config
from config.schema import GameSchema, MctsSchema, TrainSchema, NetworkSchema

cs = ConfigStore.instance()
cs.store(name="hydra_job_config", group="hydra.job", node={"chdir": True})


@define
class BaseConfig:
    game: GameSchema
    mcts: MctsSchema
    networks: NetworkSchema
    training: TrainSchema
    defaults: list[Any] = [
        {"game": MISSING},
        {"mcts": MISSING},
        {"networks": MISSING},
        {"training": MISSING},
        {"hydra.job": "hydra_job_config"},
    ]


cs.store(name="base_config", node=BaseConfig)


@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(cfg: DictConfig):
    config.populate_config(cfg)
    config.save_source_code()


if __name__ == "__main__":
    main()
