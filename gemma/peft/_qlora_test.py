import torch
from gemma.peft import quantize_lora_params

def test_quantize_lora_params():
    lora_params = {
        "dense": {
            "lora": {
                "a": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "b": torch.tensor([5.0, 6.0]),
            }
        }
    }

    quantized_params = quantize_lora_params(lora_params, bit_width=4)

    # Check that the structure remains unchanged
    assert "dense" in quantized_params
    assert "lora" in quantized_params["dense"]
    assert "a" in quantized_params["dense"]["lora"]
    assert "b" in quantized_params["dense"]["lora"]

    # Check that tensors are converted to Int8Params (from bitsandbytes)
    assert isinstance(quantized_params["dense"]["lora"]["a"], torch.Tensor)
    assert isinstance(quantized_params["dense"]["lora"]["b"], torch.Tensor)

    print("test_quantize_lora_params passed!")

if __name__ == "__main__":
    test_quantize_lora_params()
