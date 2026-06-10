# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer for DiffusionGemma."""

import dataclasses

import flax.linen as nn
from gemma.gm.nn.gemma4 import _config
from gemma.gm.nn.gemma4 import _layers
from gemma.gm.nn.gemma4 import _modules
from gemma.gm.nn.gemma4 import _transformer
from gemma.gm.utils import _dtype_params
from gemma.gm.utils import _jax_utils
from gemma.gm.vision import _token_utils
import jax.numpy as jnp
from kauldron.ktyping import Bool, Float, Int, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member

Embeddings = Float['*B L D']
Logits = Float['*B L V']


@dataclasses.dataclass(frozen=True, kw_only=True)
class SelfConditioningConfig:
  """Configuration for SelfConditioning.

  Attributes:
    features: The embedding dimension (d_model) of the transformer.
    hidden_dim: The hidden dimension used in the feed-forward block.
  """

  features: int
  hidden_dim: int

  def make(self) -> 'SelfConditioning':
    return SelfConditioning(
        features=self.features,
        hidden_dim=self.hidden_dim,
    )


class SelfConditioning(nn.Module):
  """Self-conditioning using a feed-forward block."""

  features: int
  hidden_dim: int

  def setup(self):
    self.pre_norm = _layers.RMSNorm()
    self.ffw = _modules.FeedForward(
        features=self.features,
        hidden_dim=self.hidden_dim,
    )
    self.post_norm = _layers.RMSNorm(with_scale=False)

  @typechecked
  def __call__(
      self,
      *,
      canvas_embeddings: Embeddings,
      self_conditioning_signal: Embeddings,
  ) -> Embeddings:
    normed = self.pre_norm(self_conditioning_signal)
    sc_signal = self.ffw(normed)
    combined = canvas_embeddings + sc_signal
    result = self.post_norm(combined)
    return result


class DiffusionMixin:
  """Mixin for DiffusionGemma."""

  @_jax_utils.flatten_unflatten_batch_dim()
  @typechecked
  def call_with_self_conditioning(  # pytype: disable=signature-mismatch
      self,
      tokens: Int['*B L'],
      *,
      sc_embeddings: Embeddings,
      images: UInt8['*B N H W C'] | UInt8['*B H W C'] | None = None,
      positions: Int['*B L_with_mm'] | None = None,
      cache: _config.Cache | None = None,
      attention_mask: Bool['*B L_with_mm cache_length'] | None = None,
      sliding_attention_mask: Bool['*B L_with_mm cache_length'] | None = None,
      return_last_only: bool | None = None,
      return_hidden_states: bool | None = None,
  ) -> _transformer.Output:  # Output['*B']
    """Transformer forward pass with a self-conditioning signal.

    The self-conditioning signal is passed directly as embeddings.

    Args:
      tokens: input sequence of tokens.
      sc_embeddings: embeddings from the previous denoising step.
      images: Images to feed to the vision encoder.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.
      sliding_attention_mask: transformer input mask for sliding attention.
      return_last_only: If `True`, only compute and return the logits of the
        last input token in sequence. Useful for decoding where we don't need to
        compute logits for the whole sequence, but only for the last token.
        Otherwise, return all logits. Default to `False`.
      return_hidden_states: If `True`, return the hidden states of the model.
        Otherwise, return only the logits and the cache. Default to `False`.

    Returns:
      An Output containing logits, cache, and optionally hidden_states.
    """
    if not isinstance(self, _transformer.Transformer):
      raise TypeError(
          'call_with_self_conditioning must be called on a Transformer'
          ' instance.'
      )
    return_last_only = self._get_return_last_only(return_last_only)

    with _dtype_params.initialize_param_with_dtype(
        self.dtype,
        exclude=[
            # The multi-modal params are kept in float32.
            'vision_encoder',
            'embedder.mm_input_projection',
            'embedder.mm_soft_embedding_norm',
            # Skip the LoRA params
            'lora',
        ],
    ):

      inputs = self._encode_and_get_inputs(
          tokens=tokens,
          images=images,
          positions=positions,
          attention_mask=attention_mask,
          ignore_ple_tokens=True,
      )
      del positions, attention_mask

      # Set the block-local sliding attention mask for LOCAL_SLIDING layers.
      if sliding_attention_mask is not None:
        inputs = inputs.replace(sliding_attention_mask=sliding_attention_mask)

      # In the first denoising step, `sc_signal` should be all zeros.
      is_zero_sc = jnp.all(sc_embeddings == 0.0)
      sc_signal = jnp.where(
          is_zero_sc,
          jnp.zeros_like(inputs.embeddings),
          sc_embeddings.astype(inputs.embeddings.dtype),
      )
      sc_output = self.self_conditioner(
          canvas_embeddings=inputs.embeddings,
          self_conditioning_signal=sc_signal,
      )
      inputs = inputs.replace(embeddings=sc_output)

      x, new_cache = self._apply_attention(inputs, cache)

    if return_last_only:
      last_input_token_idx = jnp.sum(inputs.inputs_mask, axis=-1) - 1
      # TODO(epot): Use `jnp.take_along_axis`
      x = x[jnp.arange(len(x)), last_input_token_idx, ...]
    elif images is not None:
      # Remove the MM extra tokens inserted.
      x = _token_utils.remove_mm_logits(
          logits=x,
          tokens=tokens,
          num_tokens_per_image=self.config.vision_encoder.num_mm_tokens_per_image,  # pytype: disable=attribute-error
      )

    logits = self.embedder.decode(x)

    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap

    return _transformer.Output(
        logits=logits,
        cache=None if cache is None else new_cache,
        hidden_states=x if return_hidden_states else None,
    )
