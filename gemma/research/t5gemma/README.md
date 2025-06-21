# T5Gemma

T5Gemma (aka encoder-decoder Gemma) was proposed in a
[research paper](https://arxiv.org/abs/2504.06225) by Google. It is a family of
encoder-decoder large langauge models, developed by adapting pretrained
decoder-only models into encoder-decoder. T5Gemma includes pretrained and
instruction-tuned variants. The architecture is based on transformer
encoder-decoder design following T5, with improvements from Gemma 2: GQA, RoPE,
GeGLU activation, RMSNorm, and interleaved local/global attention.

T5Gemma has two groups of model sizes:

* [Gemma 2](https://ai.google.dev/gemma/docs/core/model_card_2) sizes
(2B-2B, 9B-2B, and 9B-9B), which are based on the official Gemma 2 models
(2B and 9B);

* [T5](https://arxiv.org/abs/1910.10683) sizes (Small, Base, Large, and XL),
where are pretrained under the Gemma 2 framework following T5 configuration.
In addition, we also provide a model at ML size (medium large, ~2B in total),
which is in-between T5 Large and T5 XL.

This codebase provides Jax/Flax implementation of T5Gemma.

## Basic Usage

```
import kagglehub
from gemma.gm import ckpts
from gemma.research import t5gemma

# kaggle login (to get checkpoints)
kagglehub.login()

# t5gemma
preset = t5gemma.T5GemmaPreset.GEMMA2_2B_2B

t5gemma_model = preset.config.make('transformer')
t5gemma_ckpt = preset.get_checkpoint_from_kaggle(
    t5gemma.CKPTType.IT,
    t5gemma.PretrainType.PREFIXLM,
)

t5gemma_params = ckpts.load_params(t5gemma_ckpt)

# sampling
sampler = t5gemma.Sampler(
  model=t5gemma_model,
  params=t5gemma_params,
  tokenizer=preset.tokenizer
)

chat_template = '<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n'
sampler.sample(
  chat_template.format(
    user_input='Tell me an unknown interesting biology fact about the brain.'
  ),
  max_new_tokens=128
)
```

## Disclaimer

This is not an official Google product.
