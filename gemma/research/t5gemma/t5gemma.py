# Copyright 2025 DeepMind Technologies Limited.
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

"""T5 Gemma transformer."""

import dataclasses
from typing import Any, Sequence

import flax
from flax import linen as nn
from gemma.research.t5gemma import modules
import jax.numpy as jnp
from kauldron import kontext
from kauldron.typing import Array, typechecked  # pylint: disable=g-multiple-import,g-importing-member


_PADDING_ID = 0

Cache = modules.Cache
TransformerConfig = modules.TransformerConfig
TransformerOutput = modules.TransformerOutput


@flax.struct.dataclass
class T5GemmaOutput:
  """T5Gemma transformer output."""

  logits: Array['*B L V'] | Array['*B V']
  activations: Sequence[Array['*B L D'] | Array['*B D']]
  encoder_activations: Sequence[Array['*B L2 D2']] | None
  cache: Cache | None


@dataclasses.dataclass(frozen=True)
class T5GemmaConfig:
  """Configuration for the T5Gemma transformer."""

  encoder_config: TransformerConfig
  decoder_config: TransformerConfig

  def init_cache(
      self,
      batch_size: int,
      prefill_length: int,
      generation_length: int,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> Cache:
    """Initializes a new transformer cache."""
    return self.decoder_config.init_cache(
        batch_size=batch_size,
        prefill_length=prefill_length,
        generation_length=generation_length,
        dtype=dtype,
    )

  def make(
      self, name: str = 'encoder_decoder', **kwargs: Any
  ) -> 'T5Gemma':
    """Make transformer class from the configuration."""
    return T5Gemma(self, name=name, **kwargs)


class T5Gemma(nn.Module):
  """T5Gemma transformer."""

  config: T5GemmaConfig

  dtype: jnp.dtype = jnp.bfloat16

  # Keys to specify in the config which inputs to pass to the `__call__`
  # function (e.g. `tokens='batch.tokens'`).
  target_tokens: kontext.Key = kontext.REQUIRED
  input_tokens: kontext.Key = kontext.REQUIRED

  def setup(self):
    self.encoder = self.config.encoder_config.make(
        name='encoder', dtype=self.dtype
    )
    self.decoder = self.config.decoder_config.make(
        name='decoder', dtype=self.dtype
    )

  @typechecked
  def compute_encoder_activations(
      self,
      tokens: Array['B L2'],
      inputs_mask: Array['B L2'],
  ) -> TransformerOutput:
    attn_mask = make_bidirectional_attn_mask(inputs_mask)
    positions = build_positions_from_mask(inputs_mask)

    return self.encoder(
        tokens=tokens,
        positions=positions,
        self_attn_mask=attn_mask,
    )

  @typechecked
  def compute_decoder_activations(
      self,
      target_tokens: Array['B L'],
      inputs_mask: Array['B L2'],
      encoder_outputs: Array['B L2 D2'] | None = None,
      cache: Cache | None = None,
  ) -> TransformerOutput:
    targets_mask = target_tokens != _PADDING_ID
    attn_mask = make_causal_attn_mask(targets_mask)
    positions = build_positions_from_mask(targets_mask)
    cross_attn_mask = make_cross_attn_mask(
        encoder_mask=inputs_mask, decoder_mask=targets_mask
    )
    return self.decoder(
        tokens=target_tokens,
        self_attn_mask=attn_mask,
        positions=positions,
        cross_attn_kv=encoder_outputs,
        cross_attn_mask=cross_attn_mask,
        cache=cache,
    )

  @typechecked
  def __call__(
      self,
      # Decoder input tokens.
      target_tokens: Array['B L'],
      # Encoder input tokens.
      input_tokens: Array['B L2'],
      cache: Cache | None = None,
  ) -> T5GemmaOutput:
    inputs_mask = input_tokens != _PADDING_ID
    encoder_activations = self.compute_encoder_activations(
        tokens=input_tokens,
        inputs_mask=inputs_mask,
    )

    decoder_activations = self.compute_decoder_activations(
        inputs_mask=inputs_mask,
        encoder_outputs=encoder_activations.activations[-1],
        target_tokens=target_tokens,
        cache=cache,
    )
    decoder_output = self.decoder.decode(decoder_activations.activations[-1])
    return T5GemmaOutput(
        logits=decoder_output,
        activations=decoder_activations.activations,
        encoder_activations=encoder_activations.activations,
        cache=decoder_activations.cache,
    )

  @typechecked
  def decode_one_step(
      self,
      target_tokens: Array['B L'],
      positions: Array['B L'],
      cross_attn_mask: Array['B #L L2'],
      self_attn_mask: Array['#B L _'],
      cache: Cache | None = None
  ) -> T5GemmaOutput:
    decoder_activations = self.decoder(
        tokens=target_tokens,
        self_attn_mask=self_attn_mask,
        positions=positions,
        cross_attn_kv=None,
        cross_attn_mask=cross_attn_mask,
        cache=cache,
    )
    decoder_output = self.decoder.decode(decoder_activations.activations[-1])
    return T5GemmaOutput(
        logits=decoder_output,
        activations=decoder_activations.activations,
        cache=decoder_activations.cache,
        encoder_activations=None,
    )


def make_causal_attn_mask(
    input_mask: Array['B L'],
) -> Array['B L L']:
  """Make causal attention mask."""
  if len(input_mask.shape) != 2:
    raise ValueError(
        f'Input mask must be 2D (shape [B, L]), but got {input_mask.shape}.'
    )
  seq_len = input_mask.shape[-1]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
  attn_mask = input_mask[..., None, :]
  attn_mask *= causal_mask[None, ...]
  return attn_mask


def make_cross_attn_mask(
    encoder_mask: Array['B L2'],
    decoder_mask: Array['B L'],
) -> Array['B L L2']:
  """Make cross attention mask."""

  attn_mask = decoder_mask[..., jnp.newaxis] * encoder_mask[..., jnp.newaxis, :]
  return attn_mask


def make_bidirectional_attn_mask(
    input_mask: Array['B L'],
) -> Array['B L L']:
  """Make bidirectional attention mask."""
  attn_mask = input_mask[..., jnp.newaxis, :] * input_mask[..., :, jnp.newaxis]
  return attn_mask


def build_positions_from_mask(
    input_mask: Array['B L']
) -> Array['B L']:
  """Computes the `positions` from the `input_mask`."""
  positions = jnp.cumsum(input_mask, axis=-1)
  # Subtract one for all positions from the first valid one as they are
  # 0-indexed
  return positions - (positions >= 1)
