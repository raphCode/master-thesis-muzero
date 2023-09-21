from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Type, Literal, Optional, cast
from contextlib import AbstractContextManager
from collections.abc import Sequence

import torch
from torch.utils.tensorboard import SummaryWriter  # type: ignore [attr-defined]
from tensorboard.compat.proto.graph_pb2 import GraphDef  # type: ignore [import]
from tensorboard.compat.proto.node_def_pb2 import NodeDef  # type: ignore [import]
from tensorboard.compat.proto.versions_pb2 import VersionDef  # type: ignore [import]

from util import monkeypatch_wrap_args

if TYPE_CHECKING:
    from networks import Networks
    from networks.containers import NetContainer


class TensorboardLogger(AbstractContextManager["TensorboardLogger"]):
    """
    Wrapper around the tensorboard SummaryWriter, provides:
    - a custom add_graph implementation suited for the three networks
    - TBStepLogger instances to log data at specified global steps
    """

    def __init__(self, log_dir: str) -> None:
        self.sw = SummaryWriter(log_dir=log_dir)  # type: ignore [no-untyped-call]

    def add_graphs(self, nets: Networks) -> None:
        def tensor_shape_fix(shape: Sequence[int]) -> list[int]:
            # zero dimensions are problematic in tensorboard model graphs:
            # https://github.com/tensorflow/tensorboard/issues/6418
            # Replace them with -1 so that a ? shows for the zero dimension instead
            return [-1 if dim == 0 else dim for dim in shape]

        monkeypatch_wrap_args(
            torch.utils.tensorboard._proto_graph,
            "tensor_shape_proto",
            tensor_shape_fix,
        )

        # The pytorch graph parsing is a bit messy / buggy at the moment and becomes
        # easily confused, producing bogus or unwieldy graphs:
        # https://github.com/pytorch/pytorch/issues/101110
        # This happens especially when tracing the three MuZero networks together (e.g. as
        # submodules in a new Module)
        # To ensure clean graphs, combine the nodes of three individual network traces.
        # The behavior is replicated from SummaryWriter.add_graph().
        def tb_graph_nodes(cont: NetContainer) -> list[NodeDef]:
            traced_model = cont.jit()
            nodes = torch.utils.tensorboard._pytorch_graph.parse(
                traced_model.inlined_graph,
                traced_model,
                (),  # args are currently unused in the graph parsing function
            )  # type: ignore [no-untyped-call]
            return cast(list[NodeDef], nodes)

        nodes = []
        nodes += tb_graph_nodes(nets.representation)
        nodes += tb_graph_nodes(nets.prediction)
        nodes += tb_graph_nodes(nets.dynamics)

        graph = GraphDef(node=nodes, versions=VersionDef(producer=22))
        self.sw._get_file_writer().add_onnx_graph(graph)  # type: ignore [no-untyped-call]

    def add_custom_scalars_layout(
        self,
        **layouts: dict[str, tuple[Literal["Multiline"] | Literal["Margin"], list[str]]],
    ) -> None:
        self.sw.add_custom_scalars(layouts)  # type: ignore [no-untyped-call]

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
