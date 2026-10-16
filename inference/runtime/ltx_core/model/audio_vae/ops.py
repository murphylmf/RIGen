import torch
from torch import nn

from ltx_core.types import Audio

try:
    import torchaudio
except ModuleNotFoundError:                                                          
    torchaudio = None


class AudioProcessor(nn.Module):
    ''

    def __init__(
        self,
        target_sample_rate: int,
        mel_bins: int,
        mel_hop_length: int,
        n_fft: int,
    ) -> None:
        super().__init__()
        if torchaudio is None:
            raise ModuleNotFoundError("torchaudio is required to construct AudioProcessor.")
        self.target_sample_rate = target_sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=mel_hop_length,
            f_min=0.0,
            f_max=target_sample_rate / 2.0,
            n_mels=mel_bins,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        )

    def resample_audio(self, audio: Audio) -> Audio:
        ''
        if torchaudio is None:
            raise ModuleNotFoundError("torchaudio is required to resample audio.")
        if audio.sampling_rate == self.target_sample_rate:
            return audio
        resampled = torchaudio.functional.resample(audio.waveform, audio.sampling_rate, self.target_sample_rate)
        resampled = resampled.to(device=audio.waveform.device, dtype=audio.waveform.dtype)
        return Audio(waveform=resampled, sampling_rate=self.target_sample_rate)

    def waveform_to_mel(
        self,
        audio: Audio,
    ) -> torch.Tensor:
        ''
        waveform = self.resample_audio(audio).waveform

        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        mel = mel.to(device=waveform.device, dtype=waveform.dtype)
        return mel.permute(0, 1, 3, 2).contiguous()


class PerChannelStatistics(nn.Module):
    ''

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").to(x)) + self.get_buffer("mean-of-means").to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").to(x)) / self.get_buffer("std-of-means").to(x)
