# Checkpoints

## KaggleHub

Gemma models are available on KaggleHub for various formats (Jax, PyTorch,...):

*   [Gemma 3](https://www.kaggle.com/models/google/gemma-3/)
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

weights_dir = kagglehub.model_download(f'google/gemma-3/flax/gemma3-4b-it')
```

### v3

| Size | Variant     | Quantized | Name                                     |
| ---- | ----------- | --------- | ---------------------------------------- |
| 1B   | Pre-trained |           | `google/gemma-3/flax/gemma3-1b`          |
:      : (PT)        :           :                                          :
| 4B   | Pre-trained |           | `google/gemma-3/flax/gemma3-4b`          |
:      : (PT)        :           :                                          :
| 12B  | Pre-trained |           | `google/gemma-3/flax/gemma3-12b`         |
:      : (PT)        :           :                                          :
| 27B  | Pre-trained |           | `google/gemma-3/flax/gemma3-27b`         |
:      : (PT)        :           :                                          :
| 1B   | Instruction |           | `google/gemma-3/flax/gemma3-1b-it`       |
:      : Tuned (IT)  :           :                                          :
| 4B   | Instruction |           | `google/gemma-3/flax/gemma3-4b-it`       |
:      : Tuned (IT)  :           :                                          :
| 12B  | Instruction |           | `google/gemma-3/flax/gemma3-12b-it`      |
:      : Tuned (IT)  :           :                                          :
| 27B  | Instruction |           | `google/gemma-3/flax/gemma3-27b-it`      |
:      : Tuned (IT)  :           :                                          :
| 1B   | Instruction | Y         | `google/gemma-3/flax/gemma3-1b-it-int4`  |
:      : Tuned (IT)  :           :                                          :
| 4B   | Instruction | Y         | `google/gemma-3/flax/gemma3-4b-it-int4`  |
:      : Tuned (IT)  :           :                                          :
| 12B  | Instruction | Y         | `google/gemma-3/flax/gemma3-12b-it-int4` |
:      : Tuned (IT)  :           :                                          :
| 27B  | Instruction | Y         | `google/gemma-3/flax/gemma3-27b-it-int4` |
:      : Tuned (IT)  :           :                                          :

### v2

Size | Variant                | Name
---- | ---------------------- | -----------------------------------
2.6B | Pre-trained (PT)       | `google/gemma-2/flax/gemma2-2b`
9B   | Pre-trained (PT)       | `google/gemma-2/flax/gemma2-9b`
27B  | Pre-trained (PT)       | `google/gemma-2/flax/gemma2-27b`
2.6B | Instruction Tuned (IT) | `google/gemma-2/flax/gemma2-2b-it`
9B   | Instruction Tuned (IT) | `google/gemma-2/flax/gemma2-9b-it`
27B  | Instruction Tuned (IT) | `google/gemma-2/flax/gemma2-27b-it`
