import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    pass


if __name__ == "__main__":
    main()
