from typing import Any

import hydra
from attrs import define
from omegaconf import MISSING, DictConfig
from hydra.core.config_store import ConfigStore

import config
from config import C
from globals import G
from config.schema import GameSchema, MctsSchema, TrainSchema, PlayerSchema, NetworkSchema
from networks.bases import Networks

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


@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(cfg: DictConfig):
    config.populate_config(cfg)
    config.save_source_code()
    G.nets = Networks(
        representation=C.nets.factory.representation(),
        prediction=C.nets.factory.prediction(),
        dynamics=C.nets.factory.dynamics(),
        initial_latent_rep=C.nets.factory.initial_latent_rep(),
        initial_beliefs=C.nets.factory.initial_beliefs(),
    )


if __name__ == "__main__":
    main()
