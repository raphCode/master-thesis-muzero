import time
import functools
from copy import copy
from types import TracebackType
from typing import (
    Any,
    Type,
    Generic,
    Literal,
    TypeVar,
    Callable,
    Optional,
    ParamSpec,
    TypeAlias,
    cast,
)
from contextlib import AbstractContextManager, suppress, contextmanager
from collections.abc import Iterator
from typing import Any, Generic, TypeVar, Callable, Optional, ParamSpec, TypeAlias, cast
from collections.abc import Iterator
from typing import Any, Generic, TypeVar, Callable, Optional, ParamSpec, TypeAlias, cast
from collections.abc import Iterator, Sequence

import numpy as np
import torch
import gorilla  # type: ignore [import]
import numpy.typing as npt
import torch.nn.functional as F

ndarr_f32: TypeAlias = npt.NDArray[np.float32]
ndarr_f64: TypeAlias = npt.NDArray[np.float64]
ndarr_bool: TypeAlias = npt.NDArray[np.bool_]

A = TypeVar("A")
B = TypeVar("B")


def optional_map(f: Callable[[A], B]) -> Callable[[Optional[A]], Optional[B]]:
    return lambda x: f(x) if x is not None else None


class NaNWarning(RuntimeWarning):
    pass


class Sentinel:
    pass


def broadcast_cat(
    *tensors: torch.Tensor, dim: int, unsqueeze_dim: int = -1
) -> torch.Tensor:
    """
    Concatenate the tensors at the given dimension, expanding tensors as necessary.
    With this function, you can concatenate a BxCxHxW image and a BxC onehot tensor in the
    C dimension.
    """
    target_shape = list(max((len(t.shape), t.shape) for t in tensors)[1])
    target_shape[dim] = -1

    def expand(t: torch.Tensor) -> torch.Tensor:
        for _ in range(len(target_shape) - t.dim()):
            t = t.unsqueeze(unsqueeze_dim)
        return t.expand(target_shape)

    return torch.cat(list(map(expand, tensors)), dim=dim)


Fn = TypeVar("Fn", bound=Callable[..., Any])


# taken from: https://github.com/python/typing/issues/270#issuecomment-555966301
class copy_type_signature(Generic[Fn]):
    def __init__(self, target: Fn):
        pass

    def __call__(self, wrapped: Callable[..., Any]) -> Fn:
        return cast(Fn, wrapped)


class FunctionRegistry(dict[str, Callable[..., Any]]):
    """
    Acts as a decorator, collecting the wrapped functions.
    """

    def __call__(self, fn: Fn) -> Fn:
        self[fn.__name__] = fn
        return fn


@contextmanager
def hide_type_annotations(obj: Any, *annotation_names: str) -> Iterator[None]:
    """
    Context manager to temporarily delete some type annotations.
    Used to hide problematic attributes from torch.jit.script.
    """
    original_annotations = copy(obj.__annotations__)
    for name in annotation_names:
        with suppress(KeyError):
            del obj.__annotations__[name]
    yield
    obj.__annotations__ = original_annotations


def wrap_attr(obj: Any, name: str, wrapper: Callable[..., Any]) -> None:
    """
    Applies a function to an object's attribute in-place.
    """
    setattr(obj, name, wrapper(getattr(obj, name)))


def script_if_tracing(fn: Fn) -> Fn:
    """
    Typed decorator because torch.jit decorators are untyped.
    """
    return cast(Fn, torch.jit.script_if_tracing(fn))  # type: ignore [no-untyped-call]


P = ParamSpec("P")


def monkeypatch_wrap_args(obj: Any, attr: str, wrap_fn: Callable[P, Any]) -> None:
    """
    Monkeypatch a function to preprocess its arguments with a wrapper.
    """

    def wrapper(*args: P.args) -> Any:
        return original_fn(wrap_fn(*args))

    gorilla.apply(gorilla.Patch(obj, attr, wrapper, gorilla.Settings(allow_hit=True)))
    original_fn = gorilla.get_original_attribute(obj, attr)


def get_output_shape(
    module: torch.nn.Module,
    *input_shapes: Sequence[int],
    **kwargs: Any,
) -> tuple[int]:
    module.eval()
    example_inputs = [torch.zeros(1, *shape) for shape in input_shapes]
    return cast(tuple[int], tuple(module(*example_inputs, **kwargs).shape[1:]))


class TensorCache:
    """
    Creates and caches tensors to reduce memory and garbage collector pressure.
    This is a class so the caches can be dropped by deleting the instance.
    """

    def __init__(self) -> None:
        self.tensor = functools.lru_cache(maxsize=None, typed=True)(torch.tensor)
        self.zeros = functools.cache(torch.zeros)

    @functools.cache
    def onehot(self, index: int | torch.Tensor, num_classes: int) -> torch.LongTensor:
        return cast(torch.LongTensor, F.one_hot(self.tensor(index), num_classes))


T = TypeVar("T")


class RingBuffer(Generic[T]):
    """
    Storage with maximum size, append() and random access in O(1).
    New items overwrite old ones after the buffer grew to maximum size.
    """

    def __init__(self, size: int):
        self.data: list[T] = []
        self.size = size
        self.position = 0

    def append(self, value: T) -> None:
        if len(self.data) < self.size:
            self.data.append(value)
        else:
            self.data[self.position] = value
        self.position = (self.position + 1) % self.size

    def __getitem__(self, index: int) -> T:
        """
        Index 0 returns the oldest item, -1 the newest.
        """
        return self.data[(index + self.position) % len(self.data)]

    def __iter__(self) -> Iterator[T]:
        wrapped_pos = self.position % len(self.data)
        yield from self.data[wrapped_pos:] + self.data[:wrapped_pos]

    @property
    def fullness(self) -> float:
        return len(self) / self.size

    def __len__(self) -> int:
        return len(self.data)


class TimeProfiler(AbstractContextManager[None]):
    """
    Measures time to execute the context manager block as well as how many times it was called.
    """

    timings: RingBuffer[int]
    start_time: int

    def __init__(self, maxlen: int = 1000) -> None:
        self.timings = RingBuffer(maxlen)

    def __enter__(self) -> None:
        self.start_time = time.perf_counter_ns()

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> Literal[False]:
        self.timings.append(time.perf_counter_ns() - self.start_time)
        return False

    def summary(self) -> str:
        timings = np.array(self.timings)
        units = ["ns", "us", "ms", "s"]
        for unit in units:
            if np.mean(timings) < 10_000:
                break
            timings = timings / 1000
        return (
            f"{np.mean(timings):.1f} ±{np.std(timings):.1f} {unit} "
            f"{len(timings)} samples"
        )
