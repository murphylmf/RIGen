import math

import torch
import torch.nn.functional as F
from einops import rearrange


class BlurDownsample(torch.nn.Module):
    ''

    def __init__(self, dims: int, stride: int, kernel_size: int = 5) -> None:
        super().__init__()
        assert dims in (2, 3)
        assert isinstance(stride, int)
        assert stride >= 1
        assert kernel_size >= 3
        assert kernel_size % 2 == 1
        self.dims = dims
        self.stride = stride
        self.kernel_size = kernel_size

                                                                                        
                                                                                     
                                                                                                  
                                                                           
        k = torch.tensor([math.comb(kernel_size - 1, k) for k in range(kernel_size)])
        k2d = k[:, None] @ k[None, :]
        k2d = (k2d / k2d.sum()).float()                                    
        self.register_buffer("kernel", k2d[None, None, :, :])                                    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x

        if self.dims == 2:
            return self._apply_2d(x)
        else:
                                               
            b, _, f, _, _ = x.shape
            x = rearrange(x, "b c f h w -> (b f) c h w")
            x = self._apply_2d(x)
            h2, w2 = x.shape[-2:]
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f, h=h2, w=w2)
            return x

    def _apply_2d(self, x2d: torch.Tensor) -> torch.Tensor:
        c = x2d.shape[1]
        weight = self.kernel.expand(c, 1, self.kernel_size, self.kernel_size)             
        x2d = F.conv2d(x2d, weight=weight, bias=None, stride=self.stride, padding=self.kernel_size // 2, groups=c)
        return x2d
