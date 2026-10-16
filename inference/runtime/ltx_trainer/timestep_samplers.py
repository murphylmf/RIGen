import torch
from ltx_core.components.schedulers import flux_time_shift


class TimestepSampler:
    ''

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:
        ''
        raise NotImplementedError

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        ''
        raise NotImplementedError


class UniformTimestepSampler(TimestepSampler):
    ''

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:                
        return torch.rand(batch_size, device=device) * (self.max_value - self.min_value) + self.min_value

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        return self.sample(batch.shape[0], device=batch.device)


class ShiftedLogitNormalTimestepSampler(TimestepSampler):
    ''

    def __init__(self, std: float = 1.0, eps: float = 1e-3, uniform_prob: float = 0.1):
        self.std = std
        self.eps = eps
        self.uniform_prob = uniform_prob
                                                          
                                                       
                                                       
        self.normal_999_percentile = 3.0902 * std
        self.normal_005_percentile = -2.5758 * std

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        ''
        mu = self._get_shift_for_sequence_length(seq_length)

                                          
        normal_samples = torch.randn((batch_size,), device=device) * self.std + mu
        logitnormal_samples = torch.sigmoid(normal_samples)

                                                  
        percentile_999 = torch.sigmoid(torch.tensor(mu + self.normal_999_percentile, device=device))
        percentile_005 = torch.sigmoid(torch.tensor(mu + self.normal_005_percentile, device=device))

                                                                    
        zero_terminal_raw = (logitnormal_samples - percentile_005) / (percentile_999 - percentile_005)

                                                                 
        stretched_logit = torch.where(
            zero_terminal_raw >= self.eps,
            zero_terminal_raw,
            2 * self.eps - zero_terminal_raw,
        )
        stretched_logit = torch.clamp(stretched_logit, 0, 1)

                                                             
        uniform = (1 - self.eps) * torch.rand((batch_size,), device=device) + self.eps
        prob = torch.rand((batch_size,), device=device)

        return torch.where(prob > self.uniform_prob, stretched_logit, uniform)

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        ''
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, seq_length, device=batch.device)

    @staticmethod
    def _get_shift_for_sequence_length(
        seq_length: int,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> float:
                                                                                          
                                                                   
        m = (max_shift - min_shift) / (max_tokens - min_tokens)                   
        b = min_shift - m * min_tokens                         
        shift = m * seq_length + b                                    
        return shift


class TimestepShiftedUniformTimestepSampler(TimestepSampler):
    ''

    def __init__(self, shift: float, min_value: float = 0.0, max_value: float = 1.0, eps: float = 1e-6):
        self.shift = shift
        self.base_sampler = UniformTimestepSampler(min_value=min_value, max_value=max_value)
        self.eps = eps

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:                
        timesteps = self.base_sampler.sample(batch_size=batch_size, device=device).clamp(min=self.eps, max=1.0)
        shifted = [flux_time_shift(float(self.shift), 1.0, float(t)) for t in timesteps.detach().cpu().tolist()]
        return torch.tensor(shifted, dtype=timesteps.dtype, device=timesteps.device)

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        return self.sample(batch.shape[0], device=batch.device)


SAMPLERS = {
    "uniform": UniformTimestepSampler,
    "shifted_logit_normal": ShiftedLogitNormalTimestepSampler,
    "timestep_shifted_uniform": TimestepShiftedUniformTimestepSampler,
}


def example() -> None:
                                         
    import matplotlib.pyplot as plt                 

    sampler = ShiftedLogitNormalTimestepSampler()
    for seq_length in [1024, 2048, 4096, 8192]:
        samples = sampler.sample(batch_size=1_000_000, seq_length=seq_length)

                                           
        plt.hist(samples.numpy(), bins=100, density=True)
        plt.title(f"Timestep Samples for Sequence Length {seq_length}")
        plt.xlabel("Timestep")
        plt.ylabel("Density")
        plt.show()


if __name__ == "__main__":
    example()
