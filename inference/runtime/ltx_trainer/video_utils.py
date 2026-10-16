''

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from torch import Tensor


def get_video_frame_count(video_path: str | Path) -> int:
    ''
    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]

        if video_stream.frames > 0:
            return video_stream.frames

                                                                     
                                                                   
        rate = video_stream.average_rate or video_stream.base_rate
        if video_stream.duration and video_stream.time_base and rate:
            duration = Fraction(video_stream.duration) * Fraction(video_stream.time_base)
            return round(duration * Fraction(rate))

                                                     
        return sum(1 for _ in container.decode(video=0))


def read_video(video_path: str | Path, max_frames: int | None = None) -> tuple[Tensor, float]:
    ''
    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate or video_stream.base_rate or 24)

        frames = []
        for frame in container.decode(video=0):
            if max_frames is not None and len(frames) >= max_frames:
                break
            frames.append(frame.to_ndarray(format="rgb24"))

    frames_np = np.stack(frames, axis=0)                
    video = torch.from_numpy(frames_np).float().div(255.0)                          
    return video.permute(0, 3, 1, 2), fps                


def save_video(
    video_tensor: torch.Tensor,
    output_path: Path | str,
    fps: float = 24.0,
    audio: torch.Tensor | None = None,
    audio_sample_rate: int | None = None,
) -> None:
    ''
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

                                                 
    video_np = _prepare_video_array(video_tensor)
    _, height, width, _ = video_np.shape

    with av.open(str(output_path), mode="w") as container:
                            
        video_stream = container.add_stream("libx264", rate=int(fps))
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"crf": "18"}

                                      
        if audio is not None:
            if audio_sample_rate is None:
                raise ValueError("audio_sample_rate must be provided when audio is given")
            audio_stream = container.add_stream("aac", rate=audio_sample_rate)
            audio_stream.layout = "stereo"
            audio_stream.time_base = Fraction(1, audio_sample_rate)

                            
        for frame_array in video_np:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

                                 
        if audio is not None:
            _write_audio(container, audio_stream, audio, audio_sample_rate)


def _prepare_video_array(video_tensor: torch.Tensor) -> np.ndarray:
    ''
                                                
    if video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3:
        video_tensor = video_tensor.permute(1, 0, 2, 3)                                

    if not torch.isfinite(video_tensor).all():
        video_tensor = torch.nan_to_num(video_tensor, nan=0.0, posinf=1.0, neginf=0.0)

                                 
    if video_tensor.max() <= 1.0:
        video_tensor = video_tensor * 255

                                  
    return video_tensor.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()


def _write_audio(
    container: av.container.Container,
    audio_stream: av.audio.AudioStream,
    audio: torch.Tensor,
    sample_rate: int,
) -> None:
    ''
    audio = audio.cpu().float()

                                             
    if audio.ndim == 1:
        audio = audio.unsqueeze(1).repeat(1, 2)                  
    elif audio.shape[0] == 2 and audio.shape[1] != 2:
        audio = audio.T                                
    if audio.shape[1] == 1:
        audio = audio.repeat(1, 2)                  

                                                                  
    audio_int16 = (audio.clamp(-1, 1) * 32767).to(torch.int16)
    audio_interleaved = audio_int16.contiguous().view(1, -1).numpy()

                        
    frame = av.AudioFrame.from_ndarray(audio_interleaved, format="s16", layout="stereo")
    frame.sample_rate = sample_rate

                                          
    resampler = av.audio.resampler.AudioResampler(
        format=audio_stream.codec_context.format,
        layout=audio_stream.codec_context.layout,
        rate=sample_rate,
    )

    pts = 0
    for resampled_frame in resampler.resample(frame):
        resampled_frame.pts = pts
        pts += resampled_frame.samples
        for packet in audio_stream.encode(resampled_frame):
            container.mux(packet)

    for packet in audio_stream.encode():
        container.mux(packet)
