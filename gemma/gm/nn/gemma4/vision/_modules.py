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

"""Transformer sub-modules."""

from typing import Optional
from flax import linen as nn
from gemma.gm.math import _positional_embeddings
from gemma.gm.nn.gemma4._layers import ClippedEinsum
from gemma.gm.nn.gemma4._layers import Einsum
from gemma.gm.nn.gemma4._layers import RMSNorm
from gemma.gm.nn.gemma4.vision import _norms
import jax
import jax.numpy as jnp
from kauldron import kd

K_MASK = -2.3819763e38  # Set to a large negative number.
DEFAULT_ROPE_BASE_FREQUENCY = 10_000
DEFAULT_ROPE_SCALE_FACTOR = 1.0


def apply_multidimensional_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    rotary_fraction: float | None = None,
    scale_factor: float = 1.0,
) -> jax.Array:
  """Applies multidimensional RoPE.

  Args:
    inputs: Array of shape [B, L, N, H].
    positions: Array of shape [B, L, D], where D is the number of dimensions.
    base_frequency: Base frequency used to compute rotations.
    rotary_fraction: The fraction of channels to apply RoPE to. If None, apply
      to all channels.
    scale_factor: The scale factor used for positional interpolation, allowing
      an expansion of sequence length beyond the pre-trained context length.

  Returns:
    Array of shape [B, L, N, H].
  """
  if positions.ndim + 2 == inputs.ndim:
    if rotary_fraction is None or rotary_fraction == 1.0:
      return _positional_embeddings.apply_rope(
          inputs=inputs,
          positions=positions,
          base_frequency=base_frequency,
          scale_factor=scale_factor,
      )
    dim_to_rope = int(rotary_fraction * inputs.shape[-1])
    if dim_to_rope == inputs.shape[-1]:
      return _positional_embeddings.apply_rope(
          inputs=inputs,
          positions=positions,
          base_frequency=base_frequency,
          scale_factor=scale_factor,
      )
    if dim_to_rope == 0:
      return inputs
    x1 = inputs[..., :dim_to_rope]
    x2 = inputs[..., dim_to_rope:]
    x1 = _positional_embeddings.apply_rope(
        x1,
        positions=positions,
        base_frequency=base_frequency,
        scale_factor=scale_factor,
    )
    return jnp.concatenate([x1, x2], axis=-1)

  ndim = positions.shape[-1]
  num_input_channels = inputs.shape[-1]
  num_rotated_channels = num_input_channels
  if rotary_fraction is not None:
    num_rotated_channels = int(round(num_rotated_channels * rotary_fraction))
  num_rotated_channels_per_dim = 2 * (num_rotated_channels // (2 * ndim))

  assert (
      num_rotated_channels_per_dim > 0
  ), f'Requirement not satisfied: 2 * {ndim=} <= {num_input_channels=}.'

  # NOTE: because of integer rounding, num_rotated_channels is approximate.
  # The true number of rotated channels is num_rotated_channels_per_dim * ndim.

  split_points = [(k + 1) * num_rotated_channels_per_dim for k in range(ndim)]
  if rotary_fraction is None:
    split_points = split_points[:-1]  # Don't split off a non-rotated portion.
  assert all(isinstance(sp, int) for sp in split_points), split_points
  x_parts = jnp.split(inputs, split_points, axis=-1)
  y_parts = [
      _positional_embeddings.apply_rope(
          x_parts[k],
          positions=positions[..., k],
          base_frequency=base_frequency,
          scale_factor=scale_factor,
      )
      for k in range(ndim)
  ]

  if rotary_fraction is not None:  # Append the non-rotated portion.
    y_parts.append(x_parts[-1])

  return jnp.concatenate(y_parts, axis=-1)


class Attention(nn.Module):
  """Attention module."""

  num_heads: int
  num_kv_heads: int
  features: int
  head_dim: int
  rope_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
  rope_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
  attn_logits_soft_cap: float | None = None
  use_qk_norm: bool = False
  use_clipped_linears: bool = False

  def setup(self):
    linear_cls = ClippedEinsum if self.use_clipped_linears else Einsum
    self.attn_vec_einsum = linear_cls(
        shape=(self.num_heads, self.head_dim, self.features)
    )

    self.q_einsum = linear_cls(
        shape=(self.num_heads, self.features, self.head_dim)
    )
    self.kv_einsum = linear_cls(
        shape=(2, self.num_kv_heads, self.features, self.head_dim)
    )
    self.query_norm = _norms.RMSNorm()
    self.key_norm = _norms.RMSNorm()
    self.value_norm = RMSNorm(with_scale=False)
    self.attention_weights = kd.nn.Identity()

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      attn_mask: jax.Array,
  ) -> jax.Array:
    """Applies multi-head attention to the inputs.

    Args:
      x: Input sequence of shape [batch_size, seq_len, embed_dim].
      segment_pos: Input absolute positions of shape [batch_size, seq_len].
      attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].

    Returns:
      outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
    """
    query_proj = self.q_einsum('BTD,NDH->BTNH', x)
    key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)

    if self.use_qk_norm:
      query_proj = self.query_norm(query_proj)
      key_proj = self.key_norm(key_proj)

    value_proj = self.value_norm(value_proj)

    query_proj = apply_multidimensional_rope(
        query_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )
    key_proj = apply_multidimensional_rope(
        key_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )

    attn_vec = self._compute_attn_vec(
        query_proj, key_proj, value_proj, attn_mask
    )

    # [batch_size, seq_len, features]
    attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', attn_vec)
    return attn_output

  def _compute_attn_vec(
      self,
      q: jnp.ndarray,
      k: jnp.ndarray,
      v: jnp.ndarray,
      attn_mask: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Main attention computation, equivalent to the original `compute_attn_vec`."""
    b, q_len, _, h = q.shape

    # Reshape query for attention
    q = q.reshape(
        b, q_len, self.num_kv_heads, self.num_heads // self.num_kv_heads, h
    )

    out = self._qkv(q=q, k=k, v=v, attn_mask=attn_mask)
    return out

  def _qkv(
      self,
      q: jnp.ndarray,
      k: jnp.ndarray,
      v: jnp.ndarray,
      attn_mask: Optional[jnp.ndarray],
  ) -> jnp.ndarray:
    """Computes the softmax-weighted query-key-value product."""
    b, q_len, _, _, h = q.shape
    num_heads = q.shape[2] * q.shape[3]

    # Compute attention logits via einsum.
    # q: [batch, q_len, kv_heads, heads_per_kv, head_dim]
    # k: [batch, kv_len, kv_heads, head_dim]
    # logits: [batch, kv_heads, heads_per_kv, q_len, kv_len]
    attn_logits = jnp.einsum('btkgh,bskh->bkgts', q, k)

    if attn_mask is not None:
      # Add mask to logits.
      # The mask is expanded to match the logits dimensions.
      attn_logits += attn_mask[:, None, None, :, :]

    attn_weights = nn.softmax(attn_logits, axis=-1).astype(v.dtype)

    # Compute the final output via einsum.
    result = jnp.einsum('bkgts,bskh->btkgh', attn_weights, v)

    # Reshape back to the original format
    return result.reshape(b, q_len, num_heads, h)


class FeedForward(nn.Module):
  """Feed forward module."""

  features: int
  hidden_dim: int
  use_clipped_linears: bool = False

  def setup(self):
    linear_cls = ClippedEinsum if self.use_clipped_linears else Einsum
    self.gating_einsum = linear_cls(shape=(2, self.hidden_dim, self.features))
    self.linear = linear_cls(shape=(self.hidden_dim, self.features))

  def __call__(self, x):
    """Applies the feed forward module.

    Args:
      x: Input sequence of shape [batch_size, seq_len, features].

    Returns:
      Output sequence of shape [batch_size, seq_len, features].
    """

    # [batch_size, seq_len, 2, hidden_dim]
    gate = self.gating_einsum('btd,cfd->btcf', x)

    # [batch_size, seq_len, hidden_dim]
    activations = nn.gelu(gate[..., 0, :]) * gate[..., 1, :]

    return self.linear('btf,fd->btd', activations)


class Block(nn.Module):
  """Transformer block."""

  num_heads: int
  num_kv_heads: int
  embed_dim: int
  head_dim: int
  hidden_dim: int
  rope_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
  rope_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
  attn_logits_soft_cap: float | None = None
  use_qk_norm: bool = False
  use_clipped_linears: bool = False

  def setup(self):
    self.pre_attention_norm = _norms.RMSNorm()
    self.attn = Attention(
        num_heads=self.num_heads,
        features=self.embed_dim,
        head_dim=self.head_dim,
        num_kv_heads=self.num_kv_heads,
        rope_base_frequency=self.rope_base_frequency,
        rope_scale_factor=self.rope_scale_factor,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        use_qk_norm=self.use_qk_norm,
        use_clipped_linears=self.use_clipped_linears,
    )
    self.post_attention_norm = _norms.RMSNorm()
    self.pre_ffw_norm = _norms.RMSNorm()
    self.mlp = FeedForward(
        features=self.embed_dim,
        hidden_dim=self.hidden_dim,
        use_clipped_linears=self.use_clipped_linears,
    )
    self.post_ffw_norm = _norms.RMSNorm()

  def __call__(
      self,
      inputs: jax.Array,
      positions: jax.Array,
      attn_mask: jax.Array,
  ) -> jax.Array:
    """Applies the block to the inputs.

    Args:
      inputs: Input sequence of shape [batch_size, seq_len, embed_dim].
      positions: Input absolute positions of shape [batch_size, seq_len].
      attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].

    Returns:
      outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
    """
    normed_inputs = self.pre_attention_norm(inputs)
    attn_output = self.attn(
        x=normed_inputs,
        segment_pos=positions,
        attn_mask=attn_mask,
    )
    attn_output = self.post_attention_norm(attn_output)
    attn_output += inputs
    outputs = self.pre_ffw_norm(attn_output)
    outputs = self.mlp(outputs)
    outputs = self.post_ffw_norm(outputs)
    outputs += attn_output
    return outputs
