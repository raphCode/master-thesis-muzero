import torch
from trajectory import PlayerType
import torch.nn.functional as F
from attrs import Factory, define
import attrs
from torch.utils.tensorboard import SummaryWriter

from config import config as C
from trajectory import TrainingData

tensor_factory = Factory(lambda: torch.tensor(0.0))


@define
class Losses:
    latent: torch.Tensor = tensor_factory
    value: torch.Tensor = tensor_factory
    reward: torch.Tensor = tensor_factory
    policy: torch.Tensor = tensor_factory
    beliefs: torch.Tensor = tensor_factory
    player_type: torch.Tensor = tensor_factory


def process_batch(batch: list[TrainingData], sw: SummaryWriter, n: int):
    C.nets.dynamics.train()
    C.nets.representation.train()
    # TODO: move tensors to GPU

    losses = Losses()
    data_count=0
    fit_count=0
    obs_count=0
    cosine_target=torch.ones( C.train.batch_num_games)

    first = batch[0]
    obs_latent_rep, obs_beliefs = C.nets.representation(*first.observation, first. beliefs)
    latent_rep = obs_latent_rep.where(first.is_observation.unsqueeze(-1), first.latent_rep)
    beliefs = obs_beliefs.where(first.is_observation.unsqueeze(-1), first.beliefs)

    for step in batch:
        if not step.is_data.any():
            break

        if step.is_observation.any() and step is not first:
            obs_latent_rep, obs_beliefs = C.nets.representation(*step.observation, step. beliefs)
            losses.latent+=F.mse_loss(latent_rep, obs_latent_rep, reduction='none').mean(dim=1).masked_select(step.is_observation).sum()
            # TODO: remove beliefs loss, does work in the general case (or prove otherwise)
            if C.train.loss_weights.beliefs > 0:
                losses.beliefs+=F.mse_loss(beliefs, obs_beliefs, reduction='none').mean(dim=1).masked_select(step.is_observation).sum()
            obs_count += step.is_observation.count_nonzero()

        latent_rep, beliefs, reward = C.nets.dynamics(latent_rep, beliefs, step.action_onehot)
        losses.reward+=F.mse_loss(reward, step.reward, reduction='none')[step.is_data].sum()

        fit_data = step.is_data & (step.player_type != PlayerType.Terminal)
        fit_count += fit_data.count_nonzero()
        data_count += step.is_data.count_nonzero()

        value, policy_logits, player_type_logits = C.nets.prediction(latent_rep, beliefs, logits=True)
        losses.value+=F.mse_loss(value, step.value_target, reduction='none')[fit_data].sum()
        losses.policy+=F.cross_entropy(policy_logits, step.target_policy, reduction='none')[fit_data].sum()
        losses.player_type+=F.cross_entropy(player_type_logits, step.player_type, reduction='none')[step.is_data].sum()

    if obs_count > 0:
        losses.latent /= obs_count
        losses.beliefs /= obs_count
    if fit_count > 0:
        losses.value /= fit_count
        losses.policy /= fit_count
    losses.reward /= data_count
    losses.player_type /= data_count

    for k, loss in attrs.asdict(losses).items():
        sw.add_scalar(f"loss/{k}", loss, n)

    loss = (
        C.train.loss_weights.latent * losses.latent
        + C.train.loss_weights.value * losses.value
        + C.train.loss_weights.reward * losses.reward
        + C.train.loss_weights.policy * losses.policy
        + C.train.loss_weights.beliefs * losses.beliefs
        + C.train.loss_weights.player_type * losses.player_type
    )
    C.train.optimizer.zero_grad()
    loss.backward()
    #TODO: multiple param groups
    sw.add_scalar("max grad", max(p.grad.abs().max() for p in C.train.optimizer.param_groups[0]['params']), n)
    #torch.nn.utils.clip_grad_value_(C.train.optimizer.param_groups[0]['params'], 1)
    C.train.optimizer.step()

    sw.add_scalar("loss/total", loss, n)

    return loss
