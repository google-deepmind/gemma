# Core Model & Training Utilities (HD Package)

This package contains the core neural network architectures, custom Flax
modules, SFT-specific training losses, and checkpoint formatting utilities.

## Package Structure

*   `hd_gemma_network.py`: Flax wrappers around the Gemma backbone to adapt
    it for diffusion and localized prefilling.
*   `hd_gemma_ar_state_handler.py`: Gemma-specific state handler
    (`GemmaARStateHandler`) that manages the prefill, token appending, and
    positional tracking for Gemma's right-pad convention.
*   `lora.py`: Parameter-Efficient Fine-Tuning (PEFT) wrappers implementing LoRA
    adapters for JAX/Flax modules.
*   `sft_model.py`: The main Kauldron model wrapper (`SFTDiffusion`) defining
    the forward pass, prefix prefilling, autoregressive diffusion sampling
    steps, and custom losses (notably the `EncoderARLoss` for
    the sequence encoder).
*   `gemma_checkpointer.py`: Formatting evaluator that splits and saves trained
    parameters (including LoRA adapters) back into the original Gemma-compatible
    format for easy inference reloading.
*   `mask_helpers.py`: Centralized mask-building, positional tracking, and
    KV-cache manipulation utilities. Essential for managing Gemma's right-pad
    memory layout across hybrid autoregressive-diffusion workflows. It provides
    tools for causal prefilling (`make_causal_prefill_mask`,
    `build_positions_from_mask`), appending newly generated canvas tokens
    (`make_causal_attention_mask_right_pad`), enforcing block-causal boundaries
    during diffusion denoising (`create_decoder_attention_mask`), and managing
    the active cache write cursor (`set_cache_end_index`).

## Core Design: Hybrid AR-Diffusion

The core model (`SFTDiffusion` in `sft_model.py`) implements a hybrid
autoregressive-diffusion process:

1.  **AR Prefill (`sft_encode`)**: The prompt and context are encoded using
    standard causal encoder prefilling, generating a KV cache.
2.  **Localized Diffusion (`sft_decode`)**: A specific target "canvas" chunk is
    selected, corrupted with categorical noise, and denoised iteratively
    conditioned on the prefilled KV cache and positional offsets.

## Validation

All core architectures, custom layers, losses, and utilities are fully covered
by unit tests:

```bash
pytest gemma/diffusion/hackable_diffusion_adapter/hd/...
```
