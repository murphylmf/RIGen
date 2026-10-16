from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

import torch

from ltx_pipelines.utils.helpers import cleanup_memory

_M = TypeVar("_M", bound=torch.nn.Module)


@contextmanager
def gpu_model(model: _M) -> Iterator[_M]:
    ''
    try:
        yield model
    finally:
        torch.cuda.synchronize()
                                                                            
                                                 
        model.to("meta")
        cleanup_memory()
