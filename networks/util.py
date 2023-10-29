from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from torch import nn


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
