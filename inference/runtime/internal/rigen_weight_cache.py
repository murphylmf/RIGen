''
from copy import copy
from typing import Any, Callable


class EvaluationWeightCache:
    def __init__(self, loader: Callable[..., tuple[Any, Any, int]]) -> None:
        self.loader = loader
        self.key: tuple[tuple[str, str], ...] | None = None
        self.bundle: tuple[Any, Any, int] | None = None
        self.loads = 0
        self.hits = 0

    def __call__(self, **kwargs: Any) -> tuple[Any, Any, int]:
        key = tuple(sorted((name, str(value)) for name, value in kwargs.items()))
        if self.bundle is None:
            self.bundle = self.loader(**kwargs)
            self.key = key
            self.loads += 1
        elif key != self.key:
            raise ValueError("Refusing to reuse weights across different checkpoints/loading settings")
        else:
            self.hits += 1
        components, embeddings_processor, button_dim = self.bundle
                                                                                 
                                                                                 
        return copy(components), embeddings_processor, button_dim
