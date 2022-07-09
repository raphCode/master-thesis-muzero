import functools
from types import SimpleNamespace

import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from hydra.utils import get_method, instantiate

import games
from networks.bases import DynamicsNet, PredictionNet, RepresentationNet

__all__ = ["config", "populate_config"]

# global configuration container
config = SimpleNamespace()


def populate_config(hydra_cfg: DictConfig):
    C = config
    to_cont = functools.partial(OmegaConf.to_container, resolve=True)

    # GAME namespace
    C.game = SimpleNamespace()
    C.game.instance = instantiate(hydra_cfg.game.instance)
    assert isinstance(C.game.instance, games.bases.Game)
    C.game.is_teammate = get_method(hydra_cfg.game.is_teammate)
    C.game.calculate_reward = get_method(hydra_cfg.game.calculate_reward)

    # MCTS namespace
    C.mcts = SimpleNamespace(**hydra_cfg.mcts)
    C.mcts.get_node_action = get_method(hydra_cfg.mcts.get_node_action)
    C.mcts.get_node_target_policy = get_method(hydra_cfg.mcts.get_node_target_policy)
    C.mcts.get_node_selection_score = get_method(hydra_cfg.mcts.get_node_selection_score)

    # NETS namespace
    C.nets = SimpleNamespace()
    C.nets.initial_beliefs = torch.full(**to_cont(hydra_cfg.networks.initial_beliefs))
    C.nets.initial_latent_rep = torch.full(**to_cont(hydra_cfg.networks.initial_beliefs))

    C.nets.dynamics = instantiate(hydra_cfg.networks.dynamics)
    C.nets.prediction = instantiate(hydra_cfg.networks.prediction)
    C.nets.representation = instantiate(hydra_cfg.networks.representation)
    assert isinstance(C.nets.dynamics, DynamicsNet)
    assert isinstance(C.nets.prediction, PredictionNet)
    assert isinstance(C.nets.representation, RepresentationNet)

    # TRAIN namespace
    C.train = SimpleNamespace(**hydra_cfg.training)
