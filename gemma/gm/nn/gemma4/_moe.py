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

"""Moe module."""

from flax import linen as nn
from gemma.gm.nn.gemma4 import _layers
import jax
import jax.numpy as jnp


def _renormalization_factor(
    router_probs: jax.Array, choices: jax.Array
):
  """Computes the renormalization factor for routing weights."""
  indicator = jax.nn.one_hot(
      choices, router_probs.shape[-1], dtype=router_probs.dtype
  ).sum(axis=-2)
  gate_weights = indicator * router_probs
  renormalization_factor = jnp.sum(gate_weights, axis=-1, keepdims=True)
  # If sum is 0 (e.g. for padding), use 1.0 to avoid division by zero.
  return jnp.where(
      renormalization_factor > 0.0, renormalization_factor, 1.0
  )


def _expert_dispatch(
    x: jax.Array,
    expert_choices: jax.Array,
    expert_weights: jax.Array,
):
  """Sorts tokens by expert for each expert choice."""
  num_groups, group_size, k = expert_choices.shape
  x = x.reshape((-1, x.shape[-1]))
  batch_size = num_groups * group_size
  assert (
      batch_size == x.shape[0]
  ), f'batch_size ({batch_size}) must equal x.shape[0] ({x.shape[0]})'
  num_experts = expert_weights.shape[2]

  expert_choices_flat = expert_choices.ravel()  # [G * S * K]
  xs_order = expert_choices_flat.argsort()
  xs_reverse_argsort = xs_order.argsort()
  xs_indices = jnp.repeat(jnp.arange(batch_size), k)[xs_order]
  sorted_xs = x[xs_indices, :]
  expert_choices_oh = jax.nn.one_hot(
      expert_choices, num_classes=num_experts, dtype=jnp.int32
  )  # [G, S, K, E]
  xs_tokens_per_expert = jnp.sum(expert_choices_oh, axis=(0, 1, 2))  # [E]
  xs_combine_weights = (
      (
          expert_choices_oh[:, :, :, :num_experts].astype(jnp.float32)
          * jnp.expand_dims(expert_weights, axis=2)
      )
      .sum(axis=-1)
      .astype(expert_weights.dtype)
  )
  return (
      xs_tokens_per_expert,
      sorted_xs,
      xs_reverse_argsort,
      xs_combine_weights,
  )


def _expert_collect(
    sorted_xs: jax.Array,
    xs_reverse_argsort: jax.Array,
    xs_combine_weights: jax.Array,
) -> jax.Array:
  """Unshuffles tokens back to their original token order and reshapes."""
  num_groups, group_size, k = xs_combine_weights.shape
  xs = sorted_xs[xs_reverse_argsort]
  xs_reshaped = xs.reshape((num_groups, group_size, k, -1))
  return xs_reshaped


class MoE(nn.Module):
  """Mixture of Experts (MoE) module with top-k routing.

  This module implements a sparse Mixture of Experts layer where each input
  token is routed to a subset of experts (top-k) based on learned router
  logits. The outputs from the selected experts are combined using normalized
  routing weights.

  The routing algorithm:
    1. Compute router logits for each token-expert pair
    2. Apply softmax to get routing probabilities
    3. Select top-k experts per token using approximate max-k
    4. Renormalize weights among selected experts
    5. Dispatch tokens to experts, compute expert outputs, and combine

  Attributes:
    features: The input and output feature dimension (d_model).
    hidden_dim: The hidden dimension of the feed-forward network within
      each expert.
    num_experts: Total number of experts in the mixture.
    num_experts_per_datapoint: Number of experts to route each token to
      (k in top-k routing).

  Example usage:
    ```
    moe = MoE(
        features=512,
        hidden_dim=2048,
        num_experts=8,
        num_experts_per_datapoint=2,
    )
    x = jnp.ones((batch_size, seq_len, 512))
    output = moe(x)  # shape: (batch_size, seq_len, 512)
    ```
  """

  features: int
  hidden_dim: int
  num_experts: int
  num_experts_per_datapoint: int

  def setup(self):
    self.router_logits = _layers.Einsum(
        shape=(self.features, self.num_experts)
    )  # 'gsd,de->gse'
    self.linear = _layers.Einsum(
        shape=(self.num_experts, self.hidden_dim, self.features)
    )  # 'gecf,efd->gecd'
    self.gating_einsum = _layers.Einsum(
        shape=(self.num_experts, 2, self.hidden_dim, self.features)
    )  # 'gecd,ekfd->kgecf' k = 2
    self.per_expert_scale = self.param(
        'per_expert_scale',
        nn.initializers.ones,
        (self.num_experts,),
    )
    self.router_scale = self.param(
        'router_scale',
        nn.initializers.ones,
        (self.features,),
    )
    self.router_norm = _layers.RMSNorm(with_scale=False)

  def _router(
      self, router_logits: jax.Array
  ):
    router_logits = router_logits.astype(jnp.float32)
    router_probs = jax.nn.softmax(router_logits, axis=-1)
    _, choices = jax.lax.approx_max_k(
        router_logits,
        k=self.num_experts_per_datapoint,
    )  # choices: [g, s, k]
    weights = router_probs / _renormalization_factor(router_probs, choices)

    return weights, choices

  def _run_ffw_and_routing(
      self,
      x: jax.Array,
      expert_choices: jax.Array,
      expert_weights: jax.Array,
  ):
    """Runs the FFW and Routing."""
    (
        xs_tokens_per_expert,  # [E] count of tokens per expert
        sorted_xs,  # [B D], where B=k*G*S.
        xs_reverse_argsort,
        xs_combine_weights,
    ) = _expert_dispatch(x, expert_choices, expert_weights)

    group_ends = jnp.cumsum(xs_tokens_per_expert[:self.num_experts])
    broadcast_sorted_xs = jnp.repeat(
        jnp.expand_dims(sorted_xs, (0, 1)), self.num_experts, axis=1
    )  # [1, E, B, D].
    x1, x2 = self.gating_einsum('gecd,ekfd->kgecf', broadcast_sorted_xs)
    activation = nn.gelu(x1) * x2
    expert_outputs = self.linear('gecf,efd->gecd', activation)  # [1, E, B, D]
    expert_outputs = jnp.squeeze(expert_outputs, axis=0)  # [E, B, D]

    # Apply per-expert output scaling.
    # per_expert_scale shape: [E] -> [E, 1, 1] for broadcasting with [E, B, D]
    per_expert = self.per_expert_scale.astype(expert_outputs.dtype)
    expert_outputs = expert_outputs * per_expert[:, None, None]

    # Combine expert outputs. Each token in `sorted_xs` was processed by a
    # single expert. We use masks to select the output from the correct expert
    # for each token.
    ar = jnp.arange(sorted_xs.shape[0])  # [B]
    group_starts = jnp.concatenate([jnp.array([0]), group_ends[:-1]])  # [E]
    masks = (ar[None, :] >= group_starts[:, None]) & (
        ar[None, :] < group_ends[:, None]
    )  # [E, B]
    masks = jnp.expand_dims(masks, axis=-1)  # [E, B, 1]
    out = jnp.sum(expert_outputs * masks, axis=0)  # [B, D]
    out = _expert_collect(
        out,
        xs_reverse_argsort=xs_reverse_argsort,
        xs_combine_weights=xs_combine_weights,
    )
    out = jnp.einsum(
        'blkd,blk->bld',
        out,
        xs_combine_weights,
        preferred_element_type=out.dtype,
    )
    return out

  def __call__(self, x):
    """Applies the MoE module."""
    # Router: RMS-norm, scale, then compute logits.
    router_input = self.router_norm(x)
    root_size = jax.lax.rsqrt(
        jnp.array(self.features, dtype=router_input.dtype)
    )
    router_input = (
        router_input * root_size * self.router_scale.astype(router_input.dtype)
    )
    logits = self.router_logits('gsd,de->gse', router_input)
    weights, choices = self._router(logits)
    out = self._run_ffw_and_routing(x, choices, weights)
    return out


class _Weight(nn.Module):
  """Layer that provides access to a weight parameter."""

  shape: tuple[int, ...]
  initializer: nn.initializers.Initializer = nn.initializers.normal()
  weight_name: str = 'w'
  dtype: jnp.dtype | None = None

  @nn.compact
  def __call__(self):
    w = self.param(self.weight_name, self.initializer, self.shape, self.dtype)
    if isinstance(w, dict) and 'w' in w:
      w = w['w']
    return w


class MoERagged(nn.Module):
  """Mixture of Experts (MoE) module with top-k routing using ragged_dot.

  See MoE for details. This class uses jax.lax.ragged_dot for improved
  performance by avoiding dense computations with masking.
  """

  features: int
  hidden_dim: int
  num_experts: int
  num_experts_per_datapoint: int

  def setup(self):
    self.router_logits = _layers.Einsum(
        shape=(self.features, self.num_experts)
    )  # 'gsd,de->gse'

    # Weights for ragged operations.
    # Note: These are initialized to match the shapes in MoE but we will
    # process them to fit ragged_dot expectations.
    # MoE gating: (E, 2, H, F) -> used as [E, F, 2H] in ragged_dot
    self.gating_einsum = _Weight(
        shape=(self.num_experts, 2, self.hidden_dim, self.features)
    )
    # MoE linear: (E, H, F) -> used as [E, H, F] in ragged_dot
    self.linear = _Weight(
        shape=(self.num_experts, self.hidden_dim, self.features)
    )
    self.per_expert_scale = self.param(
        'per_expert_scale',
        nn.initializers.ones,
        (self.num_experts,),
    )
    self.router_scale = self.param(
        'router_scale',
        nn.initializers.ones,
        (self.features,),
    )
    self.router_norm = _layers.RMSNorm(with_scale=False)

  def _router(self, router_logits: jax.Array):
    router_logits = router_logits.astype(jnp.float32)
    router_probs = jax.nn.softmax(router_logits, axis=-1)
    _, choices = jax.lax.approx_max_k(
        router_logits,
        k=self.num_experts_per_datapoint,
    )  # choices: [g, s, k]
    weights = router_probs / _renormalization_factor(router_probs, choices)

    return weights, choices

  def _run_ffw_and_routing(
      self,
      x: jax.Array,
      expert_choices: jax.Array,
      expert_weights: jax.Array,
  ):
    """Runs the FFW and Routing using ragged_dot."""
    (
        xs_tokens_per_expert,
        sorted_xs,
        xs_reverse_argsort,
        xs_combine_weights,
    ) = _expert_dispatch(x, expert_choices, expert_weights)

    # Prepare gating weights: [E, 2, H, F] -> [E, F, 2, H] -> [E, F, 2*H]
    w_gate = self.gating_einsum()
    w_gate = jnp.transpose(w_gate, (0, 3, 1, 2))
    w_gate = w_gate.reshape(
        self.num_experts, self.features, 2 * self.hidden_dim
    )

    # Ragged dot for gating
    # lhs: [B_total, F]
    # rhs: [E, F, 2*H]
    # group_sizes: [E]
    # output: [B_total, 2*H]
    gate_out = jax.lax.ragged_dot(
        sorted_xs, w_gate, group_sizes=xs_tokens_per_expert
    )

    # Split and Apply GELU
    gate_out = gate_out.reshape(gate_out.shape[0], 2, self.hidden_dim)
    x1 = gate_out[:, 0, :]
    x2 = gate_out[:, 1, :]
    activation = nn.gelu(x1) * x2  # [B_total, H]

    # Ragged dot for linear projection
    # w_linear is [E, H, F]
    # activation is [B_total, H]
    # Output is [B_total, F]
    expert_outputs = jax.lax.ragged_dot(
        activation, self.linear(), group_sizes=xs_tokens_per_expert
    )

    # Apply per-expert output scaling.
    # Map each token to its expert index and scale accordingly.
    expert_indices = jnp.repeat(
        jnp.arange(self.num_experts),
        xs_tokens_per_expert,
        total_repeat_length=expert_outputs.shape[0],
    )
    per_expert = self.per_expert_scale.astype(expert_outputs.dtype)
    expert_outputs = expert_outputs * per_expert[expert_indices, None]

    # Collect outputs
    out = _expert_collect(
        expert_outputs,
        xs_reverse_argsort=xs_reverse_argsort,
        xs_combine_weights=xs_combine_weights,
    )

    out = jnp.einsum(
        'blkd,blk->bld',
        out,
        xs_combine_weights,
        preferred_element_type=out.dtype,
    )
    return out

  def __call__(self, x):
    """Applies the MoERagged module."""
    # Router: RMS-norm, scale, then compute logits.
    router_input = self.router_norm(x)
    root_size = jax.lax.rsqrt(
        jnp.array(self.features, dtype=router_input.dtype)
    )
    router_input = (
        router_input * root_size * self.router_scale.astype(router_input.dtype)
    )
    logits = self.router_logits('gsd,de->gse', router_input)
    weights, choices = self._router(logits)
    out = self._run_ffw_and_routing(x, choices, weights)
    return out
