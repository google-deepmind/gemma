# Checkpoints

## KaggleHub

Gemma models are available on KaggleHub for various formats (Jax, PyTorch,...):

*   [Gemma 2](https://www.kaggle.com/models/google/gemma-2/)
*   [Gemma 1](https://www.kaggle.com/models/google/gemma/)

To manually download the model:

*   Select one of the **Flax** model variations
*   Click the ⤓ button to download the model archive, then extract it locally

The archive contains both the model weights and the tokenizer, like:

```
2b/              # Directory containing model weights
tokenizer.model  # Tokenizer
```

To programmatically download the model:

```python
import kagglehub

kagglehub.login()

weights_dir = kagglehub.model_download(f'google/gemma-2/flax/gemma2-2b-it')
```

### v2

Size | Variant                | Name
---- | ---------------------- | -----------------------------------
2.6B | Pre-trained (PT)       | `google/gemma-2/flax/gemma2-2b`
9B   | Pre-trained (PT)       | `google/gemma-2/flax/gemma2-9b`
27B  | Pre-trained (PT)       | `google/gemma-2/flax/gemma2-27b`
2.6B | Instruction Tuned (IT) | `google/gemma-2/flax/gemma2-2b-it`
9B   | Instruction Tuned (IT) | `google/gemma-2/flax/gemma2-9b-it`
27B  | Instruction Tuned (IT) | `google/gemma-2/flax/gemma2-27b-it`
