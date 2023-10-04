
import logging
from functools import cache, partial

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from functools import cache
from itertools import chain
import matplotlib.animation as animation
from torch.distributions.dirichlet import Dirichlet
import cvxpy as cp


def get_angle_bisector(gradlist):
    grads = torch.stack(gradlist)
    shape = grads.shape[1:]
    unit_grads = F.normalize(grads.flatten(start_dim=1))
    unit_grads = unit_grads.reshape(-1, *shape)
    return unit_grads.sum(dim=0)

@cache
def get_nash_problem(n):
    alpha = cp.Variable(shape=(n,), nonneg=True)
    G = cp.Parameter( shape=(n,n))
    G_alpha = G @ alpha
    constraints = []
    for i in range(n):
        constraints.append(
            -cp.log(alpha[i])
            - cp.log(G_alpha[i])
            <= 0
        )

    obj = cp.Minimize( cp.sum(G_alpha))
    return alpha, G, cp.Problem(obj, constraints)


def hack(layer, name="weight", label:str=""):
    log = logging.getLogger(__name__+f": {label}.{name}")

    log.info("Register hack")
    weight = getattr(layer, name)
    delattr(layer, name)
    setattr(layer, name + "_orig", weight)

    gradlist = []

    def nash_solve(gtg):
        alpha, G, prob = get_nash_problem(len(gtg))
        G.value = gtg.numpy() 
        try:
            error = prob.solve(solver=cp.CLARABEL, warm_start=True)
        except cp.SolverError:
            pass
        return alpha.value, prob.solution.status

    def backward_hook_master_nash(_):
        n = len(gradlist)
        #log.info(f"Grad hack: {n} gradients for " + label)
        if n < 2:
            return None

        G = torch.stack(gradlist, dim=0).flatten(start_dim=1).to(dtype=torch.float64)
        GTG = G @ G.T
        alpha, status=nash_solve(GTG)
        if status != "optimal":
            log.warn(f"Solution {status} ({n} gradients)")
            return None

        assert alpha is not None
        grads = torch.stack(gradlist, dim=-1)
        alpha = torch.tensor(alpha, dtype=torch.float32)
        return torch.sum(alpha * grads, dim=-1)

    def backward_hook_clone(grad):
        # print("clone backward hook fired")
        gradlist.append(grad)

    def forward_hook(module, x):
        # print("forward hook fired")
        #assert layer is module
        gradlist.clear()
        clone = weight.view_as(weight)
        clone.register_hook(backward_hook_clone)
        setattr(module, name, clone)

    layer.register_forward_pre_hook(forward_hook)
    weight.register_hook(backward_hook_master_nash)

