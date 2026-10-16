import math
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.model.video_vae.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ltx_trainer.action_binding import is_crossfps_fixed_button_mode
from ltx_trainer.quantization import QuantizationOptions
from ltx_trainer.training_strategies.ar_sliding_ref import ARSlidingRefConfig
from ltx_trainer.training_strategies.ar_teacher import ARTeacherConfig
from ltx_trainer.training_strategies.base_strategy import TrainingStrategyConfigBase
from ltx_trainer.training_strategies.bidir_prompt_schedule import BidirPromptScheduleConfig
from ltx_trainer.training_strategies.phase_aware import PhaseAwareConfig
from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig
from ltx_trainer.training_strategies.video_to_video import VideoToVideoConfig


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(ConfigBaseModel):
    ''

    model_path: str | Path = Field(
        ...,
        description="Model path - local path to safetensors checkpoint file",
    )

    text_encoder_path: str | Path | None = Field(
        default=None,
        description="Path to text encoder (required for LTX-2/Gemma models, optional for LTXV/T5 models)",
    )

    training_mode: Literal["lora", "full"] = Field(
        default="lora",
        description="Training mode - either LoRA fine-tuning or full model fine-tuning",
    )

    attention_backend: Literal[
        "default",
        "pytorch",
        "xformers",
        "flash_attention_2",
        "flash_attention_3",
        "flash_attention_4",
    ] = Field(
        default="default",
        description=(
            "Transformer attention backend. Explicit flash_attention_2/3/4 selections are fail-closed: "
            "missing kernels and attention masks raise instead of falling back to another backend."
        ),
    )

    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to a checkpoint file or directory to load from. "
        "If a directory is provided, the latest checkpoint will be used.",
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str | Path) -> str | Path:
        ''
        is_url = str(v).startswith(("http://", "https://"))

        if is_url:
            raise ValueError(f"Model path cannot be a URL: {v}")

        if not Path(v).exists():
            raise ValueError(f"Model path does not exist: {v}")

        return v


class LoraConfig(ConfigBaseModel):
    ''

    rank: int = Field(
        default=64,
        description="Rank of LoRA adaptation",
        ge=2,
    )

    alpha: int = Field(
        default=64,
        description="Alpha scaling factor for LoRA",
        ge=1,
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout probability for LoRA layers",
        ge=0.0,
        le=1.0,
    )

    target_modules: list[str] = Field(
        default=["to_k", "to_q", "to_v", "to_out.0"],
        description="List of modules to target with LoRA",
    )

    target_block_indices: list[int] | None = Field(
        default=None,
        description=(
            "Optional transformer block indices for LoRA injection. "
            "When set, PEFT receives layers_to_transform for mid/late block-only adapters."
        ),
    )

    layers_pattern: str | None = Field(
        default="transformer_blocks",
        description="Optional PEFT layers_pattern used with target_block_indices.",
    )


def _get_strategy_discriminator(v: dict | TrainingStrategyConfigBase) -> str:
    ''
    if isinstance(v, dict):
        return v.get("name", "text_to_video")
    return v.name


                                                        
TrainingStrategyConfig = Annotated[
    Annotated[TextToVideoConfig, Tag("text_to_video")]
    | Annotated[VideoToVideoConfig, Tag("video_to_video")]
    | Annotated[PhaseAwareConfig, Tag("phase_aware")]
    | Annotated[BidirPromptScheduleConfig, Tag("bidir_prompt_schedule")]
    | Annotated[ARTeacherConfig, Tag("ar_teacher")]
    | Annotated[ARSlidingRefConfig, Tag("ar_sliding_ref")],
    Discriminator(_get_strategy_discriminator),
]


class OptimizationConfig(ConfigBaseModel):
    ''

    learning_rate: float = Field(
        default=5e-4,
        description="Learning rate for optimization",
    )

    lora_lr: float | None = Field(
        default=None,
        description="Optional learning rate for LoRA parameters in action_plus_lora mode.",
    )

    action_lr: float | None = Field(
        default=None,
        description="Optional learning rate for action-specific parameters in action-conditioned training.",
    )

    backbone_lr: float | None = Field(
        default=None,
        description="Optional explicit learning rate annotation for backbone params (useful for audits).",
    )

    text_encoder_lr: float | None = Field(
        default=None,
        description="Optional explicit learning rate annotation for the text encoder (useful for audits).",
    )

    vae_lr: float | None = Field(
        default=None,
        description="Optional explicit learning rate annotation for VAE params (useful for audits).",
    )

    steps: int = Field(
        default=3000,
        description="Number of training steps",
    )

    batch_size: int = Field(
        default=2,
        description="Batch size for training",
    )

    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of steps to accumulate gradients",
    )

    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum gradient norm for clipping",
    )

    optimizer_type: Literal["adamw", "adamw8bit"] = Field(
        default="adamw",
        description="Type of optimizer to use for training",
    )

    scheduler_type: Literal[
        "constant",
        "linear",
        "cosine",
        "warmup_cosine",
        "cosine_with_restarts",
        "polynomial",
        "step",
    ] = Field(
        default="linear",
        description="Type of scheduler to use for training",
    )

    scheduler_params: dict = Field(
        default_factory=dict,
        description="Parameters for the scheduler",
    )

    enable_gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to save memory at the cost of slower training",
    )

    startup_sanity: bool = Field(
        default=False,
        description="Run one startup-only preflight audit before training begins.",
    )

    trainable_mode: Literal["all", "action_plus_lora"] = Field(
        default="all",
        description=(
            "Which transformer parameters to train. `action_plus_lora` is for LoRA mode with "
            "action-specific base modules unfrozen."
        ),
    )

    freeze_families: list[str] = Field(
        default_factory=list,
        description="Named parameter families to keep frozen during full fine-tuning.",
    )


class AccelerationConfig(ConfigBaseModel):
    ''

    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = Field(
        default="bf16",
        description="Mixed precision training mode",
    )

    quantization: QuantizationOptions | None = Field(
        default=None,
        description="Quantization precision to use",
    )

    load_text_encoder_in_8bit: bool = Field(
        default=False,
        description="Whether to load the text encoder in 8-bit precision to save memory",
    )


class OnlineEncodeTilingConfig(ConfigBaseModel):
    spatial_tile: int = 512
    temporal_tile: int = 16
    spatial_overlap: int = 0
    temporal_overlap: int = 0

    def to_tiling_config(self) -> TilingConfig:
        return TilingConfig(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=self.spatial_tile,
                tile_overlap_in_pixels=self.spatial_overlap,
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=self.temporal_tile,
                tile_overlap_in_frames=self.temporal_overlap,
            ),
        )


class OnlineEncodeConfig(ConfigBaseModel):
    enabled: bool = False
    tiling: OnlineEncodeTilingConfig = Field(default_factory=OnlineEncodeTilingConfig)

    @model_validator(mode="after")
    def validate_tiling(self) -> "OnlineEncodeConfig":
        if not self.enabled:
            return self

        try:
            self.tiling.to_tiling_config()
        except ValueError as exc:
            raise ValueError(f"data.online_encode.tiling is invalid: {exc}") from exc

        effective_spatial_overlap = max(self.tiling.spatial_overlap, 64)
        if effective_spatial_overlap >= self.tiling.spatial_tile:
            raise ValueError(
                "data.online_encode.tiling is invalid: "
                f"spatial_tile must be greater than the effective overlap {effective_spatial_overlap} "
                f"(configured spatial_overlap={self.tiling.spatial_overlap}, encoder minimum is 64)"
            )

        effective_temporal_overlap = max(self.tiling.temporal_overlap, 16)
        if effective_temporal_overlap >= self.tiling.temporal_tile:
            raise ValueError(
                "data.online_encode.tiling is invalid: "
                f"temporal_tile must be greater than the effective overlap {effective_temporal_overlap} "
                f"(configured temporal_overlap={self.tiling.temporal_overlap}, encoder minimum is 16)"
            )

        return self


class FixedClipConfig(ConfigBaseModel):
    width: int = Field(default=832, ge=32)
    height: int = Field(default=480, ge=32)
    num_frames: int = Field(default=121, ge=1)
    fps: float = Field(default=24.0, gt=0.0)
    audio_sample_rate: int = Field(default=16000, gt=0)
    prompt_fields: list[str] = Field(
        default_factory=lambda: [
            "summary_with_audio_seed2_timestamp",
            "summary_with_audio_timestamp",
            "summary_with_audio_seed2",
            "summary_with_audio",
            "caption_o",
        ]
    )
    random_temporal_crop: bool = Field(
        default=True,
        description=(
            "For raw sources whose duration exceeds fixed_clip duration, "
            "draw a random window instead of resampling the whole clip (avoids frame dropping)."
        ),
    )
    min_source_duration: float | None = Field(
        default=None,
        ge=0.0,
        description="Drop samples shorter than this (seconds) during dataset discovery.",
    )

    @model_validator(mode="after")
    def validate_ltx2_compatibility(self) -> "FixedClipConfig":
        if self.width % 32 != 0:
            raise ValueError(f"fixed_clip.width ({self.width}) must be divisible by 32")
        if self.height % 32 != 0:
            raise ValueError(f"fixed_clip.height ({self.height}) must be divisible by 32")
        if self.num_frames % 8 != 1:
            raise ValueError(
                f"fixed_clip.num_frames ({self.num_frames}) must satisfy num_frames % 8 == 1 for LTX-2"
            )
        if not self.prompt_fields:
            raise ValueError("fixed_clip.prompt_fields must contain at least one candidate field")
        return self


class PromptMixConfig(ConfigBaseModel):
    enabled: bool = False
    caption_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    neutral_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    empty_prob: float = Field(default=0.2, ge=0.0, le=1.0)
    neutral_prompt: str = "a video"
    seed: int = 42

    @model_validator(mode="after")
    def validate_probabilities(self) -> "PromptMixConfig":
        total = self.caption_prob + self.neutral_prob + self.empty_prob
        if self.enabled and abs(total - 1.0) > 1e-6:
            raise ValueError(
                "data.prompt_mix probabilities must sum to 1.0 when enabled; "
                f"got caption+neutral+empty={total:.6f}"
            )
        return self


class JoystickNormalizationConfig(ConfigBaseModel):
    ''

    mode: Literal["none", "fixed_scale_to_unit"] = "none"
    scales: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    out_of_range_policy: Literal["error", "clip"] = "error"

    @field_validator("scales")
    @classmethod
    def validate_scales(cls, values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(f"joystick normalization scales must be finite and positive, got {values}")
        return values


class DataSourceConfig(ConfigBaseModel):
    kind: Literal["audio_only", "crossfps_audio_action", "composite"] = "audio_only"
    path: str
    name: str | None = Field(
        default=None,
        description="Optional logical source name (e.g. sekai, omniworld) for logging and manifests.",
    )
    weight: float = 1.0
    layout: Literal["auto", "precomputed", "raw_youtube_local"] = "auto"
    metadata_root: str | None = None
    media_root: str | None = None
    manifest_path: str | None = Field(
        default=None,
        description=(
            "Optional pre-built JSONL manifest of audited samples. "
            "When provided, dataset skips metadata_root scan and loads entries directly. "
            "Each line must contain at least: video_path, audio_path, prompt_text, "
            "source_fps, source_frames, source_duration. CrossFPS action/audio manifests "
            "must also contain action_path."
        ),
    )
    audio_loss_eligible_ids_path: str | None = Field(
        default=None,
        description=(
            "Optional newline-delimited clip-id list. For CrossFPS audio/action sources, "
            "all samples still provide audio conditioning, but only listed samples contribute "
            "to the audio denoising loss."
        ),
    )
    audio_loss_eligible_manifest_path: str | None = Field(
        default=None,
        description=(
            "Optional clean-audio JSONL manifest. For CrossFPS audio/action sources, "
            "all samples from manifest_path remain in training, but only clip IDs also "
            "present in this manifest contribute to the audio denoising loss."
        ),
    )
    joystick_normalization: JoystickNormalizationConfig = Field(
        default_factory=JoystickNormalizationConfig,
        description=(
            "Optional four-axis CrossFPS joystick conversion in channel order "
            "[j_left_x, j_left_y, j_right_x, j_right_y]. Defaults to no conversion."
        ),
    )


class DataConfig(ConfigBaseModel):
    ''

    preprocessed_data_root: str | None = Field(
        default=None,
        description="Path to folder containing preprocessed training data",
    )

    num_dataloader_workers: int = Field(
        default=2,
        description="Number of background processes for data loading (0 means synchronous loading)",
        ge=0,
    )

    dataloader_prefetch_factor: int = Field(
        default=2,
        description="Number of batches prefetched by each data-loader worker",
        ge=1,
    )

    dataloader_multiprocessing_context: Literal["fork", "spawn", "forkserver"] | None = Field(
        default=None,
        description=(
            "Optional multiprocessing context for data-loader workers. Use `spawn` for PyAV-backed datasets "
            "when forking after CUDA initialization is unsafe."
        ),
    )

    dataloader_pin_memory: bool = Field(
        default=True,
        description="Pin worker-produced CPU tensors before transferring them to the accelerator",
    )

    dataloader_persistent_workers: bool = Field(
        default=True,
        description="Keep data-loader workers alive between epochs",
    )

    sources: list[DataSourceConfig] = Field(
        default_factory=list,
        description="Unified phase-aware data sources for online encoding.",
    )

    online_encode: OnlineEncodeConfig = Field(default_factory=OnlineEncodeConfig)
    fixed_clip: FixedClipConfig = Field(default_factory=FixedClipConfig)
    use_cleaned_prompts: bool = False
    cleaned_metadata_path: str | None = None
    cleaned_prompt_key: str = "cleaned_prompt"
    cleaned_prompt_fallback: Literal["original", "error"] = "original"
    prompt_mix: PromptMixConfig = Field(default_factory=PromptMixConfig)


class ValidationConfig(ConfigBaseModel):
    ''

    use_dataset_example: bool = Field(
        default=False,
        description="Whether to build validation inputs from a fixed dataset sample instead of static prompts.",
    )

    dataset_example_index: int = Field(
        default=0,
        description="Dataset sample index to use when use_dataset_example is enabled.",
        ge=0,
    )

    dataset_use_first_frame_as_image: bool = Field(
        default=True,
        description="When using a dataset example, use its first video frame as the validation conditioning image.",
    )

    dataset_source_path: str | None = Field(
        default=None,
        description="Optional CrossFPS dataset root for dataset-backed validation examples.",
    )
    dataset_manifest_path: str | None = Field(
        default=None,
        description="Optional CrossFPS manifest for dataset-backed validation examples.",
    )
    dataset_source_name: str | None = Field(
        default=None,
        description="Optional source name for the dataset-backed validation example dataset.",
    )

    prompts: list[str] = Field(
        default_factory=list,
        description="List of prompts to use for validation",
    )

    negative_prompt: str = Field(
        default=DEFAULT_NEGATIVE_PROMPT,
        description="Negative prompt to use for validation examples",
    )

    images: list[str] | None = Field(
        default=None,
        description="List of image paths to use for validation. "
        "One image path must be provided for each validation prompt",
    )

    reference_videos: list[str] | None = Field(
        default=None,
        description="List of reference video paths to use for validation. "
        "One video path must be provided for each validation prompt",
    )

    reference_downscale_factor: int = Field(
        default=1,
        description="Downscale factor for reference videos in IC-LoRA validation. "
        "When > 1, reference videos are processed at 1/n resolution (e.g., 2 means half resolution). "
        "Must match the factor used during dataset preprocessing.",
        ge=1,
    )

    video_dims: tuple[int, int, int] = Field(
        default=(960, 544, 97),
        description="Dimensions of validation videos (width, height, frames). "
        "Width and height must be divisible by 32. Frames must satisfy frames % 8 == 1 for LTX-2.",
    )

    @field_validator("video_dims")
    @classmethod
    def validate_video_dims(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        ''
        width, height, frames = v

        if width % 32 != 0:
            raise ValueError(f"Width ({width}) must be divisible by 32")
        if height % 32 != 0:
            raise ValueError(f"Height ({height}) must be divisible by 32")
        if frames % 8 != 1:
            raise ValueError(f"Frames ({frames}) must satisfy frames % 8 == 1 for LTX-2 (e.g., 1, 9, 17, 25, ...)")

        return v

    frame_rate: float = Field(
        default=25.0,
        description="Frame rate for validation videos",
        gt=0,
    )

    seed: int = Field(
        default=42,
        description="Random seed used when sampling validation videos",
    )

    inference_steps: int | None = Field(
        default=None,
        description="Number of inference steps for validation. When unset, uses the original pipeline default "
        "detected from the base checkpoint metadata.",
        gt=0,
    )

    interval: int | None = Field(
        default=100,
        description="Number of steps between validation runs. If None, validation is disabled.",
        gt=0,
    )

    videos_per_prompt: int = Field(
        default=1,
        description="Number of videos to generate per validation prompt",
        gt=0,
    )

    guidance_scale: float | None = Field(
        default=None,
        description="Legacy shared CFG scale override. When unset, validation uses the original pipeline defaults "
        "detected from the base checkpoint metadata.",
        ge=1.0,
    )

    stg_scale: float | None = Field(
        default=None,
        description="Legacy shared STG scale override. When unset, validation uses the original per-modality "
        "pipeline defaults detected from the base checkpoint metadata.",
        ge=0.0,
    )

    stg_blocks: list[int] | None = Field(
        default=None,
        description="Legacy shared STG block override. When unset, validation uses the original per-modality "
        "pipeline defaults detected from the base checkpoint metadata.",
    )

    stg_mode: Literal["stg_av", "stg_v"] = Field(
        default="stg_av",
        description="Legacy shared STG mode. Only used when the legacy shared STG fields above are set.",
    )

    video_cfg_guidance_scale: float | None = Field(
        default=None,
        description="Video CFG guidance scale. Defaults to the original inference pipeline value for the checkpoint.",
        ge=1.0,
    )

    video_stg_guidance_scale: float | None = Field(
        default=None,
        description="Video STG guidance scale. Defaults to the original inference pipeline value for the checkpoint.",
        ge=0.0,
    )

    video_rescale_scale: float | None = Field(
        default=None,
        description="Video rescale guidance strength. Defaults to the original inference pipeline value for the checkpoint.",
        ge=0.0,
    )

    video_stg_blocks: list[int] | None = Field(
        default=None,
        description="Video STG block override. Defaults to the original inference pipeline value for the checkpoint.",
    )

    a2v_guidance_scale: float | None = Field(
        default=None,
        description="Audio-to-video modality guidance scale. Defaults to the original inference pipeline value "
        "for the checkpoint.",
        ge=1.0,
    )

    video_skip_step: int = Field(
        default=0,
        description="Video guidance skip-step override. 0 matches the original pipeline default of no skipping.",
        ge=0,
    )

    audio_cfg_guidance_scale: float | None = Field(
        default=None,
        description="Audio CFG guidance scale. Defaults to the original inference pipeline value for the checkpoint.",
        ge=1.0,
    )

    audio_stg_guidance_scale: float | None = Field(
        default=None,
        description="Audio STG guidance scale. Defaults to the original inference pipeline value for the checkpoint.",
        ge=0.0,
    )

    audio_rescale_scale: float | None = Field(
        default=None,
        description="Audio rescale guidance strength. Defaults to the original inference pipeline value for the checkpoint.",
        ge=0.0,
    )

    audio_stg_blocks: list[int] | None = Field(
        default=None,
        description="Audio STG block override. Defaults to the original inference pipeline value for the checkpoint.",
    )

    v2a_guidance_scale: float | None = Field(
        default=None,
        description="Video-to-audio modality guidance scale. Defaults to the original inference pipeline value "
        "for the checkpoint.",
        ge=1.0,
    )

    audio_skip_step: int = Field(
        default=0,
        description="Audio guidance skip-step override. 0 matches the original pipeline default of no skipping.",
        ge=0,
    )

    generate_audio: bool = Field(
        default=True,
        description="Whether to generate audio in validation samples. "
        "Independent of training strategy setting - you can generate audio "
        "in validation even when not training the audio branch.",
    )

    use_audio_modality: bool = Field(
        default=True,
        description="Whether validation denoising should include the audio modality and audio/video cross-modal "
        "guidance even when audio output itself is not decoded or saved.",
    )

    skip_initial_validation: bool = Field(
        default=False,
        description="Skip validation video sampling at step 0 (beginning of training)",
    )

    include_reference_in_output: bool = Field(
        default=False,
        description="For video-to-video training: concatenate the original reference video side-by-side "
        "with the generated output. The reference comes from the input video, not from the model's output.",
    )

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        ''
        if v is None:
            return None

        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(f"Number of images ({len(v)}) must match number of prompts ({num_prompts})")

        for image_path in v:
            if not Path(image_path).exists():
                raise ValueError(f"Image path '{image_path}' does not exist")

        return v

    @field_validator("reference_videos")
    @classmethod
    def validate_reference_videos(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        ''
        if v is None:
            return None

        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(f"Number of reference videos ({len(v)}) must match number of prompts ({num_prompts})")

        for video_path in v:
            if not Path(video_path).exists():
                raise ValueError(f"Reference video path '{video_path}' does not exist")

        return v

    @model_validator(mode="after")
    def validate_scaled_reference_dimensions(self) -> "ValidationConfig":
        ''
        if self.reference_downscale_factor > 1:
            width, height, _frames = self.video_dims

                                                                                 
            if width % self.reference_downscale_factor != 0:
                raise ValueError(
                    f"Width {width} is not evenly divisible by reference_downscale_factor "
                    f"{self.reference_downscale_factor}. Choose a downscale factor that divides {width} evenly."
                )
            if height % self.reference_downscale_factor != 0:
                raise ValueError(
                    f"Height {height} is not evenly divisible by reference_downscale_factor "
                    f"{self.reference_downscale_factor}. Choose a downscale factor that divides {height} evenly."
                )

            scaled_width = width // self.reference_downscale_factor
            scaled_height = height // self.reference_downscale_factor

                                                            
            if scaled_width % 32 != 0:
                raise ValueError(
                    f"Scaled reference width {scaled_width} (from {width} / {self.reference_downscale_factor}) "
                    f"is not divisible by 32. Choose a different downscale factor or adjust video_dims."
                )
            if scaled_height % 32 != 0:
                raise ValueError(
                    f"Scaled reference height {scaled_height} (from {height} / {self.reference_downscale_factor}) "
                    f"is not divisible by 32. Choose a different downscale factor or adjust video_dims."
                )

        return self

    @model_validator(mode="after")
    def validate_audio_output_settings(self) -> "ValidationConfig":
        if self.generate_audio and not self.use_audio_modality:
            raise ValueError("validation.generate_audio requires validation.use_audio_modality to also be true")
        return self

    def resolved_guider_params(self, model_path: str | Path) -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
        ''
        defaults = detect_params(str(model_path))
        video = replace(defaults.video_guider_params)
        audio = replace(defaults.audio_guider_params)

        if self.guidance_scale is not None:
            video = replace(video, cfg_scale=self.guidance_scale)
            audio = replace(audio, cfg_scale=self.guidance_scale)

        if self.stg_scale is not None:
            video = replace(video, stg_scale=self.stg_scale)
            audio_stg_scale = self.stg_scale if self.stg_mode == "stg_av" else 0.0
            audio = replace(audio, stg_scale=audio_stg_scale)

        if self.stg_blocks is not None:
            video = replace(video, stg_blocks=self.stg_blocks)
            audio_blocks = self.stg_blocks if self.stg_mode == "stg_av" else []
            audio = replace(audio, stg_blocks=audio_blocks)

        if self.video_cfg_guidance_scale is not None:
            video = replace(video, cfg_scale=self.video_cfg_guidance_scale)
        if self.video_stg_guidance_scale is not None:
            video = replace(video, stg_scale=self.video_stg_guidance_scale)
        if self.video_rescale_scale is not None:
            video = replace(video, rescale_scale=self.video_rescale_scale)
        if self.video_stg_blocks is not None:
            video = replace(video, stg_blocks=self.video_stg_blocks)
        if self.a2v_guidance_scale is not None:
            video = replace(video, modality_scale=self.a2v_guidance_scale)
        video = replace(video, skip_step=self.video_skip_step)

        if self.audio_cfg_guidance_scale is not None:
            audio = replace(audio, cfg_scale=self.audio_cfg_guidance_scale)
        if self.audio_stg_guidance_scale is not None:
            audio = replace(audio, stg_scale=self.audio_stg_guidance_scale)
        if self.audio_rescale_scale is not None:
            audio = replace(audio, rescale_scale=self.audio_rescale_scale)
        if self.audio_stg_blocks is not None:
            audio = replace(audio, stg_blocks=self.audio_stg_blocks)
        if self.v2a_guidance_scale is not None:
            audio = replace(audio, modality_scale=self.v2a_guidance_scale)
        audio = replace(audio, skip_step=self.audio_skip_step)

        if not self.use_audio_modality:
            video = replace(video, modality_scale=1.0)
            audio = replace(audio, cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0)

        return video, audio

    def resolved_inference_steps(self, model_path: str | Path) -> int:
        ''
        if self.inference_steps is not None:
            return self.inference_steps
        return detect_params(str(model_path)).num_inference_steps

    def needs_negative_prompt(self, model_path: str | Path) -> bool:
        ''
        video, audio = self.resolved_guider_params(model_path)
        return video.cfg_scale != 1.0 or (self.use_audio_modality and audio.cfg_scale != 1.0)


class CheckpointsConfig(ConfigBaseModel):
    ''

    interval: int | None = Field(
        default=None,
        description="Number of steps between checkpoint saves. If None, intermediate checkpoints are disabled.",
        gt=0,
    )

    keep_last_n: int = Field(
        default=1,
        description="Number of most recent checkpoints to keep. Set to -1 to keep all checkpoints.",
        ge=-1,
    )

    precision: Literal["bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Precision to use when saving checkpoint weights. Options: 'bfloat16' or 'float32'.",
    )

    no_resume: bool = Field(
        default=False,
        description="When True, ignore any saved training state and start from step 0. "
        "Model weights from load_checkpoint are still loaded, but optimizer/scheduler "
        "state and step counter are reset.",
    )

    save_training_state: Literal["full", "minimal", "off"] = Field(
        default="minimal",
        description="Save training state alongside checkpoints for resume. "
        "'full': optimizer + scheduler + RNG + step (~800MB for LoRA, much larger for full fine-tuning). "
        "'minimal': scheduler + RNG + step only (~few KB, sufficient for LoRA). "
        "'off': nothing saved, resume not possible.",
    )


class HubConfig(ConfigBaseModel):
    ''

    push_to_hub: bool = Field(default=False, description="Whether to push the model weights to the Hugging Face Hub")
    hub_model_id: str | None = Field(
        default=None, description="Hugging Face Hub repository ID (e.g., 'username/repo-name')"
    )

    @model_validator(mode="after")
    def validate_hub_config(self) -> "HubConfig":
        ''
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("hub_model_id must be specified when push_to_hub is True")
        return self


class WandbConfig(ConfigBaseModel):
    ''

    enabled: bool = Field(
        default=False,
        description="Whether to enable W&B logging",
    )

    project: str = Field(
        default="ltxv-trainer",
        description="W&B project name",
    )

    host: str = Field(
        default="https://api.wandb.ai",
        description="W&B host URL used for login",
    )

    key: str | None = Field(
        default=None,
        description="Optional W&B API key for explicit login",
    )

    entity: str | None = Field(
        default=None,
        description="W&B username or team",
    )

    run_name: str | None = Field(
        default=None,
        description="Optional W&B run name. Falls back to top-level run_name, then output_dir basename.",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Tags to add to the W&B run",
    )

    log_validation_videos: bool = Field(
        default=True,
        description="Whether to log validation videos to W&B",
    )


class VisualizationConfig(ConfigBaseModel):
    enabled: bool = True
    every_n_steps: int = 200
    output_subdir: str = "vis"
    decode_on_cpu: bool = False
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class SigmaSamplingConfig(ConfigBaseModel):
    mode: Literal["inherit", "uniform_range", "timestep_shift"] = Field(
        default="inherit",
        description="Optional sigma sampling override layered on top of the legacy timestep sampler config.",
    )
    range: tuple[float, float] | None = Field(
        default=None,
        description="Inclusive sigma range used when mode=uniform_range.",
    )
    timestep_shift: float | None = Field(
        default=None,
        description="Flux timestep shift value used when mode=timestep_shift.",
    )
    shifted_logit_normal_enabled: bool = Field(
        default=True,
        description="Explicit guard used by timestep_shift experiments to prevent stacking shifted logit-normal.",
    )
    log_every_n_steps: int = Field(
        default=20,
        ge=1,
        description="How often to log sampled sigma min/max/mean during training.",
    )

    @model_validator(mode="after")
    def validate_uniform_range(self) -> "SigmaSamplingConfig":
        if self.mode != "uniform_range":
            return self
        if self.range is None:
            raise ValueError("flow_matching.sigma_sampling.range must be set when mode=uniform_range")
        lo, hi = self.range
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"flow_matching.sigma_sampling.range must satisfy 0 <= lo <= hi <= 1, got {self.range}")
        return self

    @model_validator(mode="after")
    def validate_timestep_shift(self) -> "SigmaSamplingConfig":
        if self.mode != "timestep_shift":
            return self
        if self.timestep_shift is None:
            raise ValueError("flow_matching.sigma_sampling.timestep_shift must be set when mode=timestep_shift")
        if self.shifted_logit_normal_enabled:
            raise ValueError(
                "flow_matching.sigma_sampling.shifted_logit_normal_enabled must be false when mode=timestep_shift"
            )
        return self


class FlowMatchingConfig(ConfigBaseModel):
    ''

    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(
        default="shifted_logit_normal",
        description="Mode to use for timestep sampling",
    )

    timestep_sampling_params: dict = Field(
        default_factory=dict,
        description="Parameters for timestep sampling",
    )
    sigma_sampling: SigmaSamplingConfig = Field(default_factory=SigmaSamplingConfig)

    @model_validator(mode="after")
    def validate_sigma_sampling_stack(self) -> "FlowMatchingConfig":
        if self.sigma_sampling.mode == "timestep_shift" and self.timestep_sampling_mode == "shifted_logit_normal":
            raise ValueError("Do not stack shifted_logit_normal timestep sampling with sigma_sampling.mode=timestep_shift")
        return self


class ActionBindingButtonMeaningEmbeddingConfig(ConfigBaseModel):
    enabled: bool = True
    fusion: Literal["add", "concat_mlp"] = "add"
    null_meaning_text: str = ""
    cache_unique_phrases_per_batch: bool = True


class ActionBindingRemapConfig(ConfigBaseModel):
    enabled: bool = False
    prob: float = Field(default=0.5, ge=0.0, le=1.0)
    groups: list[list[str]] = Field(
        default_factory=lambda: [
            ["A", "B", "X", "Y", "LB", "RB", "LT", "RT"],
            ["DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT"],
        ]
    )


class ActionBindingDropoutConfig(ConfigBaseModel):
    control_prompt_dropout_prob: float = Field(default=0.05, ge=0.0, le=1.0)
    button_meaning_dropout_prob: float = Field(default=0.05, ge=0.0, le=1.0)
    button_id_dropout_prob: float = Field(default=0.20, ge=0.0, le=1.0)
    keep_existing_action_dropout: bool = True


class ActionBindingSemanticVocabConfig(ConfigBaseModel):
    path: str | None = None


class ActionBindingTimelinePromptConfig(ConfigBaseModel):
    prefix: str = "Actions:"
    active_threshold: float = Field(default=0.1, ge=0.0)
    quantize_sec: float = Field(default=0.1, gt=0.0)
    max_events: int = Field(default=24, ge=0)
    min_event_frames: int = Field(default=1, ge=1)
    point_event_max_frames: int = Field(default=2, ge=1)
    dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    idle_when_no_events: bool = False
    idle_text: str = "idle: the player holds position with no button input and maintains a steady stance."
    idle_start_sec: float = Field(default=0.0, ge=0.0)
    idle_end_sec: float | None = Field(default=None, ge=0.0)
    idle_requires_quiet_joystick: bool = True
    idle_joystick_threshold: float = Field(default=0.05, ge=0.0)
    no_button_when_joystick_active: bool = False
    no_button_joystick_text: str = "no discrete action button is pressed."


class ActionBindingConfig(ConfigBaseModel):
    enabled: bool = False
    button_conditioning_mode: Literal[
        "text_binding",
        "semantic_vocab",
        "prompt_timeline",
        "crossfps_all_buttons",
        "crossfps_unified",
        "crossfps_4action",
        "crossfps_legacy6",
    ] = "text_binding"
    schema_path: str | None = "configs/action_binding_schemas.yaml"
    use_scene_prompt: bool = False
    use_control_prompt: bool = True
    alias_sampling_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    button_meaning_embedding: ActionBindingButtonMeaningEmbeddingConfig = Field(
        default_factory=ActionBindingButtonMeaningEmbeddingConfig
    )
    semantic_vocab: ActionBindingSemanticVocabConfig = Field(default_factory=ActionBindingSemanticVocabConfig)
    timeline_prompt: ActionBindingTimelinePromptConfig = Field(default_factory=ActionBindingTimelinePromptConfig)
    remap: ActionBindingRemapConfig = Field(default_factory=ActionBindingRemapConfig)
    dropout: ActionBindingDropoutConfig = Field(default_factory=ActionBindingDropoutConfig)

    @model_validator(mode="after")
    def validate_button_conditioning_mode(self) -> "ActionBindingConfig":
        if self.button_conditioning_mode == "semantic_vocab" and not self.semantic_vocab.path:
            raise ValueError("action_binding.semantic_vocab.path is required for semantic_vocab mode")
        if is_crossfps_fixed_button_mode(self.button_conditioning_mode):
                                                                                           
                                                                                        
            self.button_meaning_embedding.enabled = False
            self.remap.enabled = False
        return self


class LtxTrainerConfig(ConfigBaseModel):
    ''

    run_name: str | None = Field(
        default=None,
        description="Human-readable experiment name used by W&B when configured.",
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig | None = Field(default=None)
    training_strategy: TrainingStrategyConfig = Field(
        default_factory=TextToVideoConfig,
        description="Training strategy configuration. Determines the training mode and its parameters.",
    )
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    action_binding: ActionBindingConfig = Field(default_factory=ActionBindingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

                           
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    output_dir: str = Field(
        default="outputs",
        description="Directory to save model outputs",
    )

    torch_checkpoint_debug: bool = Field(
        default=False,
        description="Enable PyTorch checkpoint debug diagnostics for short troubleshooting runs.",
    )

                                     
    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        ''
        return str(Path(v).expanduser().resolve())

    @model_validator(mode="after")
    def validate_strategy_compatibility(self) -> "LtxTrainerConfig":
        ''
                                                                                     
        if (
            self.training_strategy.name == "video_to_video"
            and self.validation.interval
            and not self.validation.reference_videos
        ):
            raise ValueError(
                "reference_videos must be provided in validation config when using video_to_video strategy"
            )

                                                                       
        if self.model.training_mode == "lora" and self.lora is None:
            raise ValueError("LoRA configuration must be provided when training_mode is 'lora'")

                                                                               
        if self.training_strategy.name == "video_to_video" and self.model.training_mode != "lora":
            raise ValueError("Training mode must be 'lora' when using video_to_video strategy")

        online_strategies = {"phase_aware", "bidir_prompt_schedule", "ar_teacher", "ar_sliding_ref"}
        if self.training_strategy.name in online_strategies and not self.data.sources:
            raise ValueError("data.sources must be configured when using an online-encoding strategy")

        if (
            self.training_strategy.name not in online_strategies
            and self.data.preprocessed_data_root is None
        ):
            raise ValueError("data.preprocessed_data_root must be configured for precomputed-data strategies")

        if self.training_strategy.name == "ar_sliding_ref":
            if self.model.attention_backend != "flash_attention_4":
                raise ValueError(
                    "ar_sliding_ref requires model.attention_backend='flash_attention_4': its isolated sink/ref "
                    "rows rely on FA4's fail-closed boolean-mask zero-row semantics."
                )
            if self.optimization.batch_size != 1:
                raise ValueError(
                    "ar_sliding_ref currently requires per-rank optimization.batch_size=1 so optional-reference "
                    "masks remain a single exact FA4 structural mask. Use gradient accumulation/global data parallelism."
                )
            if not self.data.online_encode.enabled:
                raise ValueError("ar_sliding_ref requires data.online_encode.enabled=true.")
            if not self.data.fixed_clip.random_temporal_crop:
                raise ValueError(
                    "ar_sliding_ref training requires data.fixed_clip.random_temporal_crop=true so target blocks "
                    "are sampled across the full long episode."
                )
            if any(source.kind != "crossfps_audio_action" for source in self.data.sources):
                raise ValueError("ar_sliding_ref currently accepts only crossfps_audio_action data sources.")
            if any(source.manifest_path is None for source in self.data.sources):
                raise ValueError("ar_sliding_ref requires an explicit audited manifest_path for every source.")
            if any(source.joystick_normalization.mode != "none" for source in self.data.sources):
                raise ValueError(
                    "ar_sliding_ref reads the long release's four joystick axes directly in [-1,1]; "
                    "source-side re-normalization must remain disabled."
                )
            if not self.action_binding.enabled or self.action_binding.button_conditioning_mode != "crossfps_legacy6":
                raise ValueError(
                    "ar_sliding_ref requires the established Xonotic crossfps_legacy6 action interface."
                )
            expected_frames = 1 + (
                self.training_strategy.history_chunks + 1
            ) * self.training_strategy.chunk_size * self.training_strategy.video_temporal_scale
            if self.data.fixed_clip.num_frames != expected_frames:
                raise ValueError(
                    "ar_sliding_ref fixed_clip.num_frames must encode one local VAE anchor plus four history "
                    f"chunks and one target chunk: expected {expected_frames}, got {self.data.fixed_clip.num_frames}."
                )

        return self
