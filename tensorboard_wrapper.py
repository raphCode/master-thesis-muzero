from __future__ import annotations

from types import TracebackType
from typing import Type, Literal, Optional
from contextlib import AbstractContextManager

import torch
from torch.utils.tensorboard import SummaryWriter  # type: ignore [attr-defined]


class TensorboardLogger(AbstractContextManager["TensorboardLogger"]):
    """
    Wrapper around the tensorboard SummaryWriter, provides:
    - TBStepLogger instances to log data at specified global steps
    """

    def __init__(self, log_dir: str) -> None:
        self.sw = SummaryWriter(log_dir=log_dir)  # type: ignore [no-untyped-call]

    def create_step_logger(self, step: int) -> TBStepLogger:
        return TBStepLogger(self.sw, step)

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> Literal[False]:
        self.sw.close()  # type: ignore [no-untyped-call]
        return False


class TBStepLogger:
    """
    Provides typed methods to log data to a tensorboard at a fixed global step counter.
    Advantages:
    - bundles a SummaryWriter and a global step, avoides having to pass two arguments
      everywhere
    - race-free data logging in a multiprocessing environment by ensuring that the global
      step counter is not mutated between related logging calls
    """

    def __init__(self, sw: SummaryWriter, step: int):
        self.sw = sw
        self.step = step

    def add_scalar(self, tag: str, value: float | torch.Tensor) -> None:
        self.sw.add_scalar(
            tag,
            value,
            global_step=self.step,
            new_style=True,
        )  # type: ignore [no-untyped-call]
