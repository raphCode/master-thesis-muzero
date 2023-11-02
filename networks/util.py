import itertools
from typing import Any, Callable

from torch import Tensor, nn


class ModuleFactory:
    """
    Helper class to return registered torch Modules.
    This is an alternative to nn.ModuleList, but looks nicer in the tensorboard
    model graph.
    """

    def __init__(
        self,
        parent_module: nn.Module,
        module_constructor: Callable[..., nn.Module],
        name: str,
    ):
        self.parent = parent_module
        self.mod = module_constructor
        self.name = name
        self.counter = itertools.count(1)

    def __call__(self, *args: Any, **kwargs: Any) -> nn.Module:
        mod = self.mod(*args, **kwargs)
        self.parent.add_module(self.name + str(next(self.counter)), mod)
        return mod


class NoOp(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x
