import hydra
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore

from config.impl import populate_config, save_source_code, monkeypatch_dictconfig
from config.schema import BaseConfig

cs = ConfigStore.instance()
cs.store(name="hydra_job_config", group="hydra.job", node={"chdir": True})
cs.store(name="base_config", node=BaseConfig)


@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(cfg: DictConfig) -> None:
    populate_config(cfg)
    save_source_code()


if __name__ == "__main__":
    monkeypatch_dictconfig()
    main()
