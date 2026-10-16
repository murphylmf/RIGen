from dataclasses import dataclass

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.quantization.fp8_cast import TRANSFORMER_LINEAR_DOWNCAST_MAP, UPCAST_DURING_INFERENCE
from ltx_core.quantization.fp8_scaled_mm import FP8_PREPARE_MODULE_OPS, FP8_TRANSPOSE_SD_OPS


@dataclass(frozen=True)
class QuantizationPolicy:
    ''

    sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = ()

    @classmethod
    def fp8_cast(cls) -> "QuantizationPolicy":
        ''
        return cls(
            sd_ops=TRANSFORMER_LINEAR_DOWNCAST_MAP,
            module_ops=(UPCAST_DURING_INFERENCE,),
        )

    @classmethod
    def fp8_scaled_mm(cls) -> "QuantizationPolicy":
        ''
        try:
            import tensorrt_llm  # noqa: F401 - importing verifies optional backend availability
        except ImportError as e:
            raise ImportError("tensorrt_llm is not installed, skipping FP8 scaled MM quantization") from e

        return cls(
            sd_ops=FP8_TRANSPOSE_SD_OPS,
            module_ops=(FP8_PREPARE_MODULE_OPS,),
        )
