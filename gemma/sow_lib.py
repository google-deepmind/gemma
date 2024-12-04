# Copyright 2024 DeepMind Technologies Limited.
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

"""Utilities for sowing intermediate activations."""

import dataclasses
import jax


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class BlockIntermediates:
  """Intermediate activations for a single layer (block)."""

  # Dense residual stream activations.
  rs_after_attention: jax.Array | None = None
  rs_after_ffw: jax.Array | None = None

  # Sparse representations for large activations.
  ffw_hidden_topk_values: jax.Array | None = None
  ffw_hidden_topk_indices: jax.Array | None = None
  attn_logits_topk_values: jax.Array | None = None
  attn_logits_topk_indices: jax.Array | None = None

  def merge(self, decoding_step, step_intermediates: 'BlockIntermediates'):
    """Merges the intermediate activations from one step."""

    # This logic is the same for all intermediates. The second dimenions is the
    # length dimension, where we want to merge the intermediates from
    # multiple steps.
    for field  in dataclasses.fields(self.__class__):
      value = getattr(self, field.name)
      if value is not None:
        step_value = getattr(step_intermediates, field.name)
        if step_value is None:
          raise ValueError(
              'Intermediate step value is None for field %s' % field.name
          )
        setattr(
            self,
            field.name,
            value.at[:, decoding_step + 1].set(step_value[:, 0, ...]),
        )

  def trim(self, max_length: int):
    """Trims the intermediate activations to the given length."""
    for field in dataclasses.fields(self.__class__):
      value = getattr(self, field.name)
      if value is not None:
        setattr(self, field.name, value[:, :max_length, ...])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TransformerIntermediates:
  """Intermediate activations of one transformer step."""

  # Embeddings of the input tokens.
  embeddings: jax.Array | None = None

  # Intermediate activations of each layer.
  layers: list[BlockIntermediates] = dataclasses.field(default_factory=list)

  def merge(
      self, decoding_step, step_intermediates: 'TransformerIntermediates'
  ):
    """Merges the intermediate activations from one step."""
    if self.embeddings is not None:
      assert step_intermediates.embeddings is not None
      self.embeddings = self.embeddings.at[:, decoding_step + 1, ...].set(
          step_intermediates.embeddings[:, 0, ...]
      )
    for layer, step_layer in zip(self.layers, step_intermediates.layers):
      layer.merge(decoding_step, step_layer)

  def trim(self, max_length: int):
    """Trims the intermediate activations to the given length."""
    if self.embeddings is not None:
      self.embeddings = self.embeddings[:, :max_length, ...]
    for layer in self.layers:
      layer.trim(max_length)


@dataclasses.dataclass(frozen=True)
class SowModule:
  """Module for sowing intermediate activations."""

  # Whether to sow embeddings.
  embeddings: bool = False

  # Whether to sow activations after each attention block (in residual stream).
  rs_after_attention: bool = False

  # Whether to sow activations after each FFW block (in residual stream).
  # This is the same as the residual stream activations after a whole layer.
  rs_after_ffw: bool = False

  # If non-zero, top-k activations in a ffw hidden layer are sowed.
  # We use a sparse representation here to save memory.
  ffw_hidden_topk: int = 0

  # If non-zero, top-k attention logits are sowed.
  # We use a sparse representation here to save memory.
  attn_logits_topk: int = 0

  def maybe_sow_embeddings(
      self,
      embeddings: jax.Array,
      intermediates: TransformerIntermediates | None,
  ):
    """Sows embeddings if configured."""
    if intermediates is not None and self.embeddings:
      intermediates.embeddings = embeddings

  def maybe_sow_rs_after_attention(
      self,
      activations: jax.Array,
      intermediates: BlockIntermediates | None,
  ):
    """Sows activations after attention if configured."""
    if intermediates is not None and self.rs_after_attention:
      intermediates.rs_after_attention = activations

  def maybe_sow_rs_after_ffw(
      self,
      activations: jax.Array,
      intermediates: BlockIntermediates | None,
  ):
    """Sows activations after FFW if configured."""
    if intermediates is not None and self.rs_after_ffw:
      intermediates.rs_after_ffw = activations

  def maybe_sow_ffw_hidden_topk(
      self,
      activations: jax.Array,
      intermediates: BlockIntermediates | None,
  ):
    """Sows top-k activations in a ffw hidden layer if configured."""
    if intermediates is not None and self.ffw_hidden_topk:
      (
          intermediates.ffw_hidden_topk_values,
          intermediates.ffw_hidden_topk_indices,
      ) = jax.lax.top_k(activations, self.ffw_hidden_topk)

  def maybe_sow_attn_logits_topk(
      self,
      logits: jax.Array,
      intermediates: BlockIntermediates | None,
  ):
    """Sows top-k attention logits if configured."""
    if intermediates is not None and self.attn_logits_topk:
      (
          intermediates.attn_logits_topk_values,
          intermediates.attn_logits_topk_indices,
      ) = jax.lax.top_k(logits, self.attn_logits_topk)
