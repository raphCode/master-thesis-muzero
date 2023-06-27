import functools
from copy import copy
from typing import Any, Generic, TypeVar, Callable, Optional, TypeAlias, cast
from contextlib import suppress, contextmanager
from collections.abc import Iterator

import numpy as np
import torch
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


Fn = TypeVar("Fn", bound=Callable[..., Any])


# taken from: https://github.com/python/typing/issues/270#issuecomment-555966301
class copy_type_signature(Generic[Fn]):
    def __init__(self, target: Fn):
        pass

    def __call__(self, wrapped: Callable[..., Any]) -> Fn:
        return cast(Fn, wrapped)


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

    def __len__(self) -> int:
        return len(self.data)
