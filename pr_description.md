# Refactor vision token utils to decouple from tokenizer

## Description
Refactors `_token_utils.py` to accept `special_tokens` as an argument instead of hardcoding `Gemma3Tokenizer`.
This addresses the TODO `This value should be propagated from the model.` and makes the utility completely model-agnostic.

## Changes
- **`gemma/gm/vision/_token_utils.py`**:
    - Updated `add_extra_tokens_for_images` to accept `special_tokens`.
    - Updated `remove_mm_logits` to accept `special_tokens`.
    - Removed hardcoded `_tokenizer.Gemma3Tokenizer` usage.
- **`gemma/gm/utils/_types.py`**: Propagate `self.config.special_tokens` to `add_extra_tokens_for_images`.
- **`gemma/gm/nn/_transformer.py`**: Propagate `self.config.input_config.special_tokens` to `remove_mm_logits`.
- **`gemma/gm/nn/gemma3n/_transformer.py`**: Propagate `self.config.input_config.special_tokens` to `remove_mm_logits`.
- **`gemma/gm/vision/_token_utils_test.py`**: Updated tests to use a mock `SpecialTokens` class.

## Verification
- Verified function signatures and basic logic via script (mocking dependencies).
- Unit tests updated to match new signature.
