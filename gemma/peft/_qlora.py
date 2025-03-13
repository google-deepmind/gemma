from typing import Dict, Any
import torch
import bitsandbytes as bnb  # Required for QLoRA

_ParamsDict = Dict[str, Any]

def quantize_lora_params(lora_params: _ParamsDict, bit_width: int = 4) -> _ParamsDict:
    """Applies quantization to LoRA parameters.

    Args:
        lora_params (_ParamsDict): LoRA parameter dictionary.
        bit_width (int): Bit-width for quantization (default: 4-bit).

    Returns:
        _ParamsDict: Quantized LoRA parameter dictionary.
    """
    if bit_width not in [4, 8]:
        raise ValueError("Only 4-bit and 8-bit quantization are supported.")

    quantized_params = {}

    for layer, params in lora_params.items():
        if isinstance(params, dict):
            quantized_params[layer] = quantize_lora_params(params, bit_width)
        else:
            dtype = torch.float16 if bit_width == 8 else torch.float32
            quantized_params[layer] = bnb.nn.Int8Params(params.to(dtype))

    return quantized_params
