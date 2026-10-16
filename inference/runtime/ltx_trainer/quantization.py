                                                     
                                                              
from typing import Literal

import torch
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from ltx_trainer import logger

QuantizationOptions = Literal[
    "int8-quanto",
    "int4-quanto",
    "int2-quanto",
    "fp8-quanto",
    "fp8uz-quanto",
]

                                       
                                                                 
                                                                                         
                                                                                      
EXCLUDE_PATTERNS = [
                                    
    "patchify_proj",
    "audio_patchify_proj",
    "action_joystick_proj",
    "*action_joy*",
    "*action_button*",
    "proj_out",
    "audio_proj_out",
                                                                              
                                                                               
    "*adaln*",
    "time_proj",
    "timestep_embedder*",
                                    
    "caption_projection*",
    "audio_caption_projection*",
                                                               
    "*norm*",
]

                                                                        
                                                               
                                                                                                    
SKIP_ROOT_MODULES = {
    "patchify_proj",
    "audio_patchify_proj",
    "action_joystick_proj",
    "proj_out",
    "audio_proj_out",
    "audio_caption_projection",
}


def quantize_model(
    model: torch.nn.Module,
    precision: QuantizationOptions,
    quantize_activations: bool = False,
    device: torch.device | str | None = None,
) -> torch.nn.Module:
    ''
    from optimum.quanto import freeze, quantize                 

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    weight_quant = _get_quanto_dtype(precision)

    if quantize_activations:
        logger.debug("Quantizing model weights and activations")
        activations_quant = weight_quant
    else:
        activations_quant = None

                                                            
    original_device = next(model.parameters()).device

                                                                           
    if hasattr(model, "transformer_blocks"):
        logger.debug("Quantizing model using block-by-block approach for memory efficiency")
        _quantize_blockwise(
            model,
            weight_quant=weight_quant,
            activations_quant=activations_quant,
            device=device,
        )
    else:
                                                 
        model.to(device)
        quantize(model, weights=weight_quant, activations=activations_quant, exclude=EXCLUDE_PATTERNS)
        freeze(model)

                                      
    model.to(original_device)

    return model


def _quantize_blockwise(
    model: torch.nn.Module,
    weight_quant: torch.dtype,
    activations_quant: torch.dtype | None,
    device: torch.device,
) -> None:
    ''
    from optimum.quanto import freeze, quantize                 

    original_dtype = next(model.parameters()).dtype
    transformer_blocks = list(model.transformer_blocks)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Quantizing transformer blocks", total=len(transformer_blocks))

        for block in transformer_blocks:
                               
            block.to(device, dtype=original_dtype, non_blocking=True)

                             
            quantize(block, weights=weight_quant, activations=activations_quant, exclude=EXCLUDE_PATTERNS)
            freeze(block)

                                                             
            block.to("cpu", non_blocking=True)

            progress.advance(task)

                                                                                               
                                                                               
    logger.debug("Quantizing remaining model components")

    for name, module in model.named_children():
        if name == "transformer_blocks":
            continue                     

        if name in SKIP_ROOT_MODULES:
            logger.debug(f"Skipping quantization for module: {name}")
            continue                                

                                                     
        module.to(device, dtype=original_dtype, non_blocking=True)
        quantize(module, weights=weight_quant, activations=activations_quant, exclude=EXCLUDE_PATTERNS)
        freeze(module)
        module.to("cpu", non_blocking=True)


def _get_quanto_dtype(precision: QuantizationOptions) -> torch.dtype:
    ''
    from optimum.quanto import (                 
        qfloat8,
        qfloat8_e4m3fnuz,
        qint2,
        qint4,
        qint8,
    )

    if precision == "int2-quanto":
        return qint2
    elif precision == "int4-quanto":
        return qint4
    elif precision == "int8-quanto":
        return qint8
    elif precision in ("fp8-quanto", "fp8uz-quanto"):
        if torch.backends.mps.is_available():
            raise ValueError("FP8 quantization is not supported on MPS devices. Use int2, int4, or int8 instead.")
        if precision == "fp8-quanto":
            return qfloat8
        elif precision == "fp8uz-quanto":
            return qfloat8_e4m3fnuz

    raise ValueError(f"Invalid quantization precision: {precision}")
