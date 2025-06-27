# Research

## Custom transformer

For experimentation, you might want to create smaller custom versions of
Transformer, rather than using the default official "Gemma" pre-trained ones.

Here is an example to create a text-only, 12 layers transformer:

```python
class MyTinyTransformer(gm.nn.Transformer):
  config: gm.nn.config.TransformerConfig = gm.nn.config.TransformerConfig(
      final_logit_softcap=None,
      num_embed=262144,  # Vocab size, matching the tokenizer
      embed_dim=896,
      hidden_dim=4 * 896,
      num_heads=4,
      head_dim=256,
      num_kv_heads=1,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      use_qk_norm=True,
      attention_types=gm.nn.config.make_attention_layers_types(
          pattern=gm.nn.config.GEMMA3_ATTENTION_PATTERN,
          num_layers=12,
      ),
      query_pre_attn_norm=gm.nn.config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
      attn_logits_soft_cap=None,
      sliding_window_size=512,
      transpose_gating_einsum=True,
      local_base_frequency=10_000,
      global_base_frequency=1_000_000,
      vision_encoder=None,  # Text-only
  )

  INFO = gm.nn.config.ModelInfo(
      tokenizer_version=3,  # Auto-select the tokenizer in the sampler
  )
```

## Gemma-related projects

The Gemma repository also contain various non-official research projects around
Gemma, located in the [research/](https://github.com/google-deepmind/gemma/tree/main/gemma/research/)
directory:

*   [`t5gemma`](https://github.com/google-deepmind/gemma/blob/main/gemma/research/t5gemma/README.md):
    Encoder/decoder Gemma architecture, based on Gemma 2.
