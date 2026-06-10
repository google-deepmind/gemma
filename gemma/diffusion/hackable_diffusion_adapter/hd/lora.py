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

"""LoRA wrapper around Gemma models."""

import dataclasses
import functools
import re
from typing import Any, Sequence
from absl import logging
from flax import linen as nn
from gemma import peft
from gemma.gm.nn import _layers
from gemma.gm.nn.gemma3n import _layers as _gemma3n_layers
from gemma.gm.nn.gemma4 import _layers as _gemma4_layers
import jax
import jax.numpy as jnp
from kauldron import kontext
import numpy as np

_SUPPORTED_MODULES = (
    nn.Dense,
    nn.Einsum,
    nn.DenseGeneral,
    _layers.Einsum,
    _gemma4_layers.Einsum,
    _gemma4_layers.ClippedEinsum,
    _gemma3n_layers.Einsum,
)

# Sentinel value mirroring HuggingFace PEFT's ``target_modules='all-linear'``.
# When used, LoRA is applied to every module in ``_SUPPORTED_MODULES``.
ALL_LINEAR = 'all-linear'


class LoRA(nn.Module):
  r"""Wrapper around a Gemma model to enable LoRA.

  The model wrapped will have all its ``nn.Dense``, ``nn.Einsum``, … layers
  replaced by their LoRA versions.  See ``gemma.peft`` documentation for more
  details.

  Usage — wrap any Gemma model and optionally restrict which modules receive
  LoRA adapters via ``target_modules``::

    # ── 1. All linear layers (default) ──────────────────────────────────
    # Every supported module (Dense, Einsum, DenseGeneral, …) gets LoRA.
    # This includes: attention projections, MLP projections, router_logits,
    # and self_conditioner/ffw.  The embedder (nn.Embed) is naturally
    # excluded since it is not a linear layer.
    lora_model = LoRA(rank=8, model=gemma_model)
    # Equivalent explicit form:
    lora_model = LoRA(rank=8, model=gemma_model, target_modules='all-linear')

    # ── 2. Attention-only LoRA ──────────────────────────────────────────
    # Target all attention projections (Q, KV / K, output).
    # Note: some layers use ``kv_einsum`` (fused K+V), others use
    # separate ``k_einsum``; include both to cover the full model.
    lora_model = LoRA(
        rank=8,
        model=gemma_model,
        target_modules=['q_einsum', 'kv_einsum', 'k_einsum', 'attn_vec_einsum'],
    )

    # ── 3. MLP-only LoRA ───────────────────────────────────────────────
    # Target all feed-forward / MLP projections (both mlp and mlp2)
    # inside transformer blocks. The ``layer_`` prefix avoids matching
    # self_conditioner/ffw.
    lora_model = LoRA(
        rank=8,
        model=gemma_model,
        target_modules=[r'layer_\d+/mlp'],
    )

    # ── 4. Attention + MLP (everything inside transformer blocks) ──────
    # Note: 'mlp' also matches 'mlp2' since it's a substring.
    lora_model = LoRA(
        rank=8,
        model=gemma_model,
        target_modules=[r'layer_\d+/(attn|mlp)'],
    )

    # ── 5. Specific layers only ────────────────────────────────────────
    # Fine-tune only layers 0 and 1:
    lora_model = LoRA(
        rank=8,
        model=gemma_model,
        target_modules=[r'layer_[01]/'],
    )
    # Fine-tune the last 4 layers (e.g. layers 26-29):
    lora_model = LoRA(
        rank=8,
        model=gemma_model,
        target_modules=[r'layer_2[6-9]/'],
    )

    # ── 6. Combining layer and module filters ──────────────────────────
    # Q and K projections in layers 0–9 only:
    lora_model = LoRA(
        rank=8,
        model=gemma_model,
        target_modules=[r'layer_[0-9]/(attn/(q_einsum|kv_einsum|k_einsum))'],
    )

    # ── 7. Query/Value only (classic LoRA recipe) ──────────────────────
    lora_model = LoRA(
        rank=8,
        model=gemma_model,
        target_modules=['q_einsum', 'kv_einsum', 'k_einsum'],
    )

    # ── 8. Router projections only (MoE layers) ───────────────────────
    lora_model = LoRA(
        rank=8,
        model=gemma_model,
        target_modules=['router_logits'],
    )

  The Gemma checkpoint parameter tree has the following structure per layer
  (flattened with ``flax.traverse_util.flatten_dict``)::

    ('layer_0', 'attn', 'q_einsum', 'w')           # query projection
    ('layer_0', 'attn', 'kv_einsum', 'w')           # fused key+value
    ('layer_0', 'attn', 'attn_vec_einsum', 'w')     # output projection
    ('layer_0', 'mlp', 'gating_einsum', 'w')        # MLP gate
    ('layer_0', 'mlp', 'linear', 'w')               # MLP down-projection
    ('layer_0', 'mlp', 'router_logits', 'w')        # MoE router
    ('layer_0', 'mlp2', 'gating_einsum')            # second MLP gate
    ('layer_0', 'mlp2', 'linear')                   # second MLP down-proj
    ('self_conditioner', 'ffw', 'gating_einsum')    # self-conditioner

  Some layers (e.g. 5, 11, 17, 23, 29) use a separate ``k_einsum`` instead
  of the fused ``kv_einsum``.

  Attributes:
    rank: The rank of the LoRA decomposition.
    model: The model to wrap.
    dtype: The dtype to use for the LoRA weights.
    verbose: If ``True``, logs diagnostic strings for the LoRA layers.
    target_modules: An optional list of module name patterns (regex) to apply
      LoRA to. Each pattern is matched via ``re.search`` against the full scope
      path of the module (components joined by ``/``, e.g.
      ``layer_0/attn/q_einsum``). A module gets LoRA if **any** pattern matches.
      When ``None`` (the default) or ``'all-linear'``, LoRA is applied to
      **all** supported modules.
  """

  _: dataclasses.KW_ONLY

  rank: int
  model: nn.Module
  dtype: jnp.dtype = jnp.bfloat16
  verbose: bool = False
  target_modules: str | Sequence[str] | None = None

  def __post_init__(self):
    """Shares scope with the wrapped model to flatten the param hierarchy."""
    super().__post_init__()
    # Share scope so the param structure is {'params': model_params}
    # instead of {'params': {'model': model_params}}.
    if self.scope is not None:
      nn.share_scope(self, self.model)

  def _lora_interceptor(self):
    """Returns the LoRA ModuleInterceptor context manager."""
    replace_module_fn = functools.partial(
        _replace_by_lora,
        rank=self.rank,
        dtype=self.dtype,
        verbose=self.verbose,
        target_modules=self.target_modules,
    )
    return peft.ModuleInterceptor(replace_module_fn)

  @nn.compact
  def __call__(self, *args, **kwargs):
    """Calls the model."""
    with self._lora_interceptor():
      return self.model(*args, **kwargs)

  @nn.compact
  def encoder_call(self, *args, **kwargs):
    """Calls the model's encoder_call with LoRA adapters active."""
    with self._lora_interceptor():
      return self.model.encoder_call(*args, **kwargs)

  @nn.compact
  def init_cache(self, *args, **kwargs):
    """Calls the model's init_cache with LoRA adapters active."""
    with self._lora_interceptor():
      return self.model.init_cache(*args, **kwargs)

  def __kontext_keys__(self) -> dict[str, str]:
    """Kauldron keys when calling `kontext.get_from_keys_obj`.

    Returns:
      A dictionary mapping attribute names to their Kauldron keypaths.
    """
    # Forward the keys from the wrapped model.
    # This allows defining the config as:
    # gm.nn.LoRA(
    #   model=MyModel(
    #     input='batch.input',  # keys propagated to the `LoRA`
    #   ),
    # )
    return kontext.get_keypaths(self.model)

  def __getattr__(self, name: str) -> Any:
    """Forwards attribute accesses to the wrapped model."""
    return getattr(self.model, name)


def _lora_debug_string(module: nn.Module) -> str | None:
  """Returns a debug string for supported LoRA modules, or None."""
  if isinstance(module, _SUPPORTED_MODULES):
    return f'[LoRA] {type(module).__name__} ({module.name}) <- {module.path}'
  else:
    return None


def _matches_target_modules(
    module: nn.Module,
    target_modules: str | Sequence[str] | None,
) -> bool:
  """Check whether ``module`` matches any of the ``target_modules`` patterns.

  Args:
    module: A Flax module with a ``.path`` attribute (available inside
      ``nn.intercept_methods``).
    target_modules: Regex patterns to match against the ``/``-joined scope path.
      ``None`` or ``'all-linear'`` means "match everything".

  Returns:
    ``True`` if the module should receive LoRA.
  """
  if target_modules is None or target_modules == ALL_LINEAR:
    return True
  if isinstance(target_modules, str):
    raise ValueError(
        f'Unsupported target_modules string: {target_modules!r}. '
        f'Use {ALL_LINEAR!r}, None, or a list of regex patterns.'
    )
  try:
    path_str = '/'.join(module.path)
  except AttributeError:
    # Module doesn't have a path (not yet bound). Fall back to name.
    path_str = module.name or ''
  return any(re.search(pattern, path_str) for pattern in target_modules)


def _replace_by_lora(
    module: nn.Module,
    *,
    rank: int,
    dtype: np.dtype,
    verbose: bool,
    target_modules: str | Sequence[str] | None = None,
) -> nn.Module:
  """Replaces compatible modules by their LoRA version."""
  if verbose:
    debug_str = _lora_debug_string(module)
    if debug_str:
      logging.info(debug_str)

  if not isinstance(module, _SUPPORTED_MODULES):
    return module

  if not _matches_target_modules(module, target_modules):
    if verbose:
      logging.info(
          '[LoRA] SKIPPED %s (%s) <- %s (no target_modules match)',
          type(module).__name__,
          module.name,
          module.path,
      )
    return module

  match module:
    case nn.Dense():
      return peft.LoRADense(rank=rank, dtype=dtype, wrapped=module)
    case nn.Einsum():
      return peft.LoRAEinsum(rank=rank, dtype=dtype, wrapped=module)
    case nn.DenseGeneral():
      return peft.LoRADenseGeneral(rank=rank, dtype=dtype, wrapped=module)
    case _:
      # All custom Einsum variants (gm.nn, gemma4, gemma3n, nano, etc.)
      # use `_LoRAEinsum` wrapper. The name hack is required because
      # FeedForward uses `nn.share_scope` to flatten two Einsum modules
      # into the same param scope — the two wrappers need distinct names.
      if module.weight_name != 'w':
        name = f'_LoRAEinsum_{module.weight_name}'
      else:
        name = None
      return _LoRAEinsum(name=name, rank=rank, dtype=dtype, wrapped=module)


class _LoRAEinsum(nn.Module):
  """LoRA wrapper around a Gemma Einsum."""

  _: dataclasses.KW_ONLY
  rank: int
  dtype: np.dtype
  wrapped: nn.Module  # Any Einsum variant (gm.nn, gemma4, gemma3n, nano)

  # Do not use `nn.share_scope` here as the `wrapped` module inside
  # `FeedForward` already uses `nn.share_scope`, so the two Einsums used in
  # the `FeedForward` would collide.

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    # Warning: Calling multiple times with different `einsum_str` will
    # fail as the decomposition would not be the same.
    adapter = peft.LoRAEinsumAdapter(
        name='lora',
        rank=self.rank,
        dtype=self.dtype,
        einsum_str=eqn,
        shape=self.wrapped.shape,
    )
    return self.wrapped(eqn, x) + adapter(x)


def fuse_lora_params(params: dict[str, Any]) -> dict[str, Any]:
  """Fuse LoRA weights into base weights, returning a base-only param tree.

  Takes a param tree that contains both base weights and LoRA adapter weights
  (``lora/a`` and ``lora/b``), and produces a new tree where each base weight
  has been replaced by ``W_base + delta_W``, with
  ``delta_W = tensordot(a, b, axes=([-1], [0]))``.

  The LoRA decomposition always stores the rank dimension as:
    * the **last** axis of ``a``
    * the **first** axis of ``b``

  so ``tensordot(a, b, ([-1], [0]))`` contracts the rank and yields a tensor
  with the same shape as the original weight.

  Args:
    params: A nested param dict (e.g. from ``model.init(...)['params']``)
      containing both base and LoRA parameters.

  Returns:
    A new nested param dict with the same structure as the base-only model
    (all LoRA keys removed, base weights updated).

  Example::

    params = lora_model.init(rng, ...)['params']
    base_params = fuse_lora_params(params)
    # base_params can now be used with the non-LoRA model.

  Raises:
    KeyError: If no base weight is found for a LoRA adapter.
  """
  import flax.traverse_util  # pylint: disable=g-import-not-at-top

  # Split into base-only and LoRA-only trees.
  original, lora_tree = peft.split_params(params)

  # Flatten both to {path_tuple: array}.
  original_flat = flax.traverse_util.flatten_dict(original)
  lora_flat = flax.traverse_util.flatten_dict(lora_tree)

  # Group LoRA leaves into (a, b) pairs keyed by their parent path.
  #  A LoRA leaf path looks like: (..., 'lora', 'a') or (..., 'lora', 'b').
  lora_pairs: dict[tuple[str, ...], dict[str, jnp.ndarray]] = {}
  for path, value in lora_flat.items():
    if len(path) >= 2 and path[-2] == 'lora' and path[-1] in ('a', 'b'):
      parent = path[:-2]  # everything above 'lora/a' or 'lora/b'
      lora_pairs.setdefault(parent, {})[path[-1]] = value

  # For each (a, b) pair, find the matching base weight and fuse.
  fused_count = 0
  for lora_parent, ab in lora_pairs.items():
    if 'a' not in ab or 'b' not in ab:
      logging.warning(
          'Incomplete LoRA pair at %s, skipping.', '/'.join(lora_parent)
      )
      continue

    a, b = ab['a'], ab['b']

    # Locate the base weight.  Two layouts exist:
    #
    # (A) Shared-scope wrappers (LoRADense, LoRAEinsum):
    #     lora_parent == base_weight_parent, e.g. both are ('layer_0','attn',
    #     'q_einsum').  The base weight is at ('layer_0','attn','q_einsum','w')
    #     or ('...', 'kernel').
    #
    # (B) Non-shared-scope wrapper (_LoRAEinsum):
    #     lora_parent is ('layer_0','mlp','_LoRAEinsum_gating_einsum'), and the
    #     base weight is at ('layer_0','mlp','gating_einsum').  The wrapper name
    #     encodes the weight_name as '_LoRAEinsum_{weight_name}'.
    #     Or lora_parent is ('...','q_einsum','_LoRAEinsum_0'), and the base
    #     weight key is ('...','q_einsum','w').

    base_key = _find_base_weight_key(lora_parent, original_flat)
    if base_key is None:
      raise KeyError(
          f'Could not find base weight for LoRA at {"/".join(lora_parent)}. '
          f'a.shape={a.shape}, b.shape={b.shape}.'
      )

    base_w = original_flat[base_key]

    delta_w = _compute_lora_delta(a, b, target_shape=base_w.shape)

    original_flat[base_key] = base_w + delta_w.astype(base_w.dtype)
    fused_count += 1

  logging.info('Fused %d LoRA adapter(s) into base weights.', fused_count)

  # Unflatten back to the original nested structure.
  return flax.traverse_util.unflatten_dict(original_flat)


def _find_base_weight_key(
    lora_parent: tuple[str, ...],
    original_flat: dict[tuple[str, ...], Any],
) -> tuple[str, ...] | None:
  """Finds the base weight key corresponding to a LoRA adapter location.

  Handles two naming conventions:

  1. **_LoRAEinsum_{weight_name}** (e.g. ``_LoRAEinsum_gating_einsum``):
     The base weight is a sibling named ``{weight_name}``.
  2. **_LoRAEinsum_0** (default name, ``weight_name='w'``):
     The base weight is a sibling named ``w``.
  3. **Shared-scope** (LoRADense, LoRAEinsum, LoRADenseGeneral):
     The ``lora`` dict is a direct child of the base weight scope.
     Look for ``kernel`` or ``w`` among siblings.

  Args:
    lora_parent: Tuple path to the LoRA adapter's parent scope (above
      ``lora/a``).
    original_flat: Flattened base-only param dict.

  Returns:
    The key in ``original_flat`` for the base weight, or ``None``.
  """
  wrapper_name = lora_parent[-1] if lora_parent else ''
  scope_parent = lora_parent[:-1]

  # Case 1: _LoRAEinsum_{weight_name} → sibling is {weight_name}
  if wrapper_name.startswith('_LoRAEinsum_'):
    suffix = wrapper_name[len('_LoRAEinsum_') :]
    if suffix != '0':
      # Named variant: weight_name is the suffix itself.
      candidate = scope_parent + (suffix,)
      if candidate in original_flat:
        return candidate

    # Default variant (_LoRAEinsum_0): weight_name is 'w'.
    candidate = scope_parent + ('w',)
    if candidate in original_flat:
      return candidate

  # Case 2: Shared-scope — lora_parent IS the base weight scope.
  # Check for common weight leaf names.
  for weight_name in ('w', 'kernel'):
    candidate = lora_parent + (weight_name,)
    if candidate in original_flat:
      return candidate

  # Case 3: The base weight might be the lora_parent path itself (rare, but
  # possible if the original module stores a bare array without a sub-key).
  if lora_parent in original_flat:
    return lora_parent

  return None


def _compute_lora_delta(
    a: jnp.ndarray,
    b: jnp.ndarray,
    *,
    target_shape: tuple[int, ...],
) -> jnp.ndarray:
  """Contract LoRA matrices ``a`` and ``b``, producing ``target_shape``.

  The LoRA decomposition stores:
    * ``a`` with shape ``(*in_dims, rank)``
    * ``b`` with shape ``(rank, *out_dims)``

  The original weight is an **interleaving** of ``in_dims`` and ``out_dims``
  (both in their original order from the einsum weight string).  This function
  finds that unique interleaving, constructs the correct ``einsum`` string, and
  raises ``ValueError`` if the interleaving is ambiguous (e.g. two axes share
  the same size).

  Args:
    a: LoRA matrix A with shape ``(*in_dims, rank)``.
    b: LoRA matrix B with shape ``(rank, *out_dims)``.
    target_shape: Shape of the original weight.

  Returns:
    The fused delta weight with shape ``target_shape``.
  """
  in_dims = a.shape[:-1]
  out_dims = b.shape[1:]

  # Fast path: if (in_dims, out_dims) already matches target, use tensordot.
  if in_dims + out_dims == target_shape:
    return jnp.tensordot(a, b, axes=([-1], [0]))

  # Find how in_dims and out_dims interleave to form target_shape.
  interleaving = _find_interleaving(target_shape, in_dims, out_dims)

  # Build einsum string:  a_labels,b_labels->target_labels
  #   a_labels: one letter per in_dim + 'r' for rank
  #   b_labels: 'r' + one letter per out_dim
  #   target_labels: letters placed according to the interleaving
  n_a = len(in_dims)
  n_b = len(out_dims)
  letters = 'abcdefghijklmnopqstuvwxyz'  # 'r' reserved for rank
  a_letters = letters[:n_a]
  b_letters = letters[n_a : n_a + n_b]

  a_str = a_letters + 'r'
  b_str = 'r' + b_letters

  target_str = ''
  ai, bi = 0, 0
  for source in interleaving:
    if source == 'a':
      target_str += a_letters[ai]
      ai += 1
    else:
      target_str += b_letters[bi]
      bi += 1

  return jnp.einsum(f'{a_str},{b_str}->{target_str}', a, b)


def _find_interleaving(
    target: tuple[int, ...],
    seq_a: tuple[int, ...],
    seq_b: tuple[int, ...],
) -> list[str]:
  """Find the unique interleaving of ``seq_a`` and ``seq_b`` that produces ``target``.

  Both ``seq_a`` and ``seq_b`` must appear in ``target`` in their original
  relative order.  Raises ``ValueError`` if zero or more-than-one valid
  interleavings exist (the latter means axis sizes are ambiguous).

  Args:
    target: The shape to decompose.
    seq_a: Sizes of the ``a`` (in) dimensions, in order.
    seq_b: Sizes of the ``b`` (out) dimensions, in order.

  Returns:
    A list of ``'a'`` / ``'b'`` tags, one per element of ``target``.
  """
  solutions: list[list[str]] = []

  def _solve(ti: int, ai: int, bi: int, path: list[str]):
    if len(solutions) > 1:
      return  # early exit -- already ambiguous
    if ti == len(target):
      if ai == len(seq_a) and bi == len(seq_b):
        solutions.append(list(path))
      return
    # Try taking the next element from seq_a.
    if ai < len(seq_a) and target[ti] == seq_a[ai]:
      path.append('a')
      _solve(ti + 1, ai + 1, bi, path)
      path.pop()
    # Try taking the next element from seq_b.
    if bi < len(seq_b) and target[ti] == seq_b[bi]:
      path.append('b')
      _solve(ti + 1, ai, bi + 1, path)
      path.pop()

  _solve(0, 0, 0, [])

  if len(solutions) == 0:
    raise ValueError(
        f'Cannot interleave a_dims={seq_a} and b_dims={seq_b} to produce '
        f'weight shape {target}.'
    )
  if len(solutions) > 1:
    raise ValueError(
        f'Ambiguous LoRA fusion: weight shape {target} can be formed by '
        f'{len(solutions)} interleavings of a_dims={seq_a} and '
        f'b_dims={seq_b}. This happens when two axes share the same size. '
        'Cannot safely determine axis mapping.'
    )
  return solutions[0]
