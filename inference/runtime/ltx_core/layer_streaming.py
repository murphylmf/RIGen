''

from __future__ import annotations

import functools
import itertools
import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    ''
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj


class _LayerStore:
    ''

    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)

                                                            
        self._pinned: list[dict[str, torch.Tensor]] = []
        self._on_gpu: set[int] = set()

        for layer in layers:
            pinned: dict[str, torch.Tensor] = {}
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                pinned_tensor = tensor.data.pin_memory()
                tensor.data = pinned_tensor
                pinned[name] = pinned_tensor
            self._pinned.append(pinned)

    def _check_idx(self, idx: int) -> None:
        if idx < 0 or idx >= self.num_layers:
            raise IndexError(f"Layer index {idx} out of range [0, {self.num_layers})")

    def is_on_gpu(self, idx: int) -> bool:
        return idx in self._on_gpu

    def move_to_gpu(self, idx: int, layer: nn.Module, *, non_blocking: bool = False) -> None:
        ''
        self._check_idx(idx)
        if idx in self._on_gpu:
            return
        pinned = self._pinned[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            param.data = pinned[name].to(self.target_device, non_blocking=non_blocking)
        self._on_gpu.add(idx)

    def evict_to_cpu(self, idx: int, layer: nn.Module) -> None:
        ''
        self._check_idx(idx)
        if idx not in self._on_gpu:
            return
        pinned = self._pinned[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            param.data = pinned[name]
        self._on_gpu.discard(idx)

    def cleanup(self) -> None:
        ''
        for pinned_dict in self._pinned:
            pinned_dict.clear()
        self._pinned.clear()


class _AsyncPrefetcher:
    ''

    def __init__(self, store: _LayerStore, layers: nn.ModuleList) -> None:
        self._store = store
        self._layers = layers
        self._stream = torch.cuda.Stream(device=store.target_device)
        self._events: dict[int, torch.cuda.Event] = {}

    def prefetch(self, idx: int) -> None:
        ''
        if self._store.is_on_gpu(idx) or idx in self._events:
            return
        with torch.cuda.stream(self._stream):
            self._store.move_to_gpu(idx, self._layers[idx], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._stream)
            self._events[idx] = event

    def wait(self, idx: int) -> None:
        ''
        event = self._events.pop(idx, None)
        if event is not None:
            torch.cuda.current_stream(self._store.target_device).wait_event(event)

    def cleanup(self) -> None:
        ''
        self._events.clear()
        self._stream = None
        self._layers = None
        self._store = None


class LayerStreamingWrapper(nn.Module):
    ''

    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        prefetch_count: int = 2,
    ) -> None:
        if prefetch_count < 1:
            raise ValueError("prefetch_count must be >= 1")
        super().__init__()
                                                                                
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
                                                                                      
        self._prefetch_count = min(prefetch_count, len(self._layers) - 1)
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        self._setup()

                                                                        
                      
                                                                        

    def _setup(self) -> None:
                                                                                    
        self._store = _LayerStore(self._layers, self._target_device)

                                                      
        layer_tensor_ids: set[int] = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)

                                                                          
        for idx in range(min(self._prefetch_count + 1, len(self._layers))):
            self._store.move_to_gpu(idx, self._layers[idx])

                                                            
        self._prefetcher = _AsyncPrefetcher(self._store, self._layers)
        self._register_hooks()

    def _register_hooks(self) -> None:
        idx_map: dict[int, int] = {id(layer): idx for idx, layer in enumerate(self._layers)}
        num_layers = len(self._layers)

        def _pre_hook(
            module: nn.Module,
            _args: Any,                
            *,
            idx: int,
        ) -> None:
                                                                             
            self._prefetcher.wait(idx)
            if not self._store.is_on_gpu(idx):
                self._store.move_to_gpu(idx, module)

                                                                            
                                                                             
                                                                              
                                                                            
                                                  
            compute_stream = torch.cuda.current_stream(self._target_device)
            for param in itertools.chain(module.parameters(), module.buffers()):
                param.data.record_stream(compute_stream)

                                                                                 
            for offset in range(1, self._prefetch_count + 1):
                self._prefetcher.prefetch((idx + offset) % num_layers)

        def _post_hook(
            module: nn.Module,
            _args: Any,                
            _output: Any,                
            *,
            idx: int,
        ) -> None:
                                                                     
            self._store.evict_to_cpu(idx, module)

        for layer in self._layers:
            idx = idx_map[id(layer)]
            h1 = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            h2 = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
            self._hooks.extend([h1, h2])

    def teardown(self) -> None:
        ''
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

                                                                              
                                                                         
                                                  
        torch.cuda.synchronize(device=self._target_device)
        if self._prefetcher is not None:
            self._prefetcher.cleanup()
            self._prefetcher = None

                                 
        for idx, layer in enumerate(self._layers):
            self._store.evict_to_cpu(idx, layer)

        for p in self._model.parameters():
            p.data = p.data.to("cpu")
        for b in self._model.buffers():
            b.data = b.data.to("cpu")

                                                                           
                                                                          
                                                                       
                                                                               
        self._store.cleanup()

                                                                        
                                      
                                                                        

    def forward(self, *args: Any, **kwargs: Any) -> Any:                
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:                
        ''
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
