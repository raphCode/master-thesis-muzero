from typing import Any

import hydra
from attrs import define
from omegaconf import MISSING, OmegaConf, DictConfig
from hydra.core.config_store import ConfigStore

import config
from config.schema import GameSchema, MctsSchema, TrainSchema, NetworkSchema

cs = ConfigStore.instance()


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
    ]


cs.store(name="base_config", node=BaseConfig)


@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(cfg: DictConfig):
    pass


if __name__ == "__main__":
    main()
