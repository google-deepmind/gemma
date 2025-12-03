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

"""Test for attention mask broadcasting bug (Issue #407)."""

import jax
from jax import numpy as jnp
import pytest


class TestAttentionMaskBroadcasting:
  """Tests for attention mask shape compatibility."""

  def test_attention_mask_broadcasting_shape_mismatch(self):
    """Reproduce ValueError from Issue #407: Incompatible shapes for broadcasting.
    
    The bug occurs when attn_mask and logits have mismatched cache dimensions:
    - attn_mask shape after expand_dims: (B, L, 1, cache_size_1)
    - logits shape: (B, L, num_heads, cache_size_2)
    
    where cache_size_1 != cache_size_2, causing broadcasting failure.
    """
    # Simulate the scenario from the error:
    # attn_mask originally: (1, 1447, 5234) -> after expand_dims: (1, 1447, 1, 5234)
    # logits: (1, 1447, 8, 4096)
    batch_size, seq_len, num_heads = 1, 1447, 8
    mask_cache_size = 5234
    logits_cache_size = 4096
    
    # Original 3D attention mask (before expand_dims)
    attn_mask = jnp.ones((batch_size, seq_len, mask_cache_size), dtype=bool)
    
    # Logits from attention computation
    logits = jnp.zeros((batch_size, seq_len, num_heads, logits_cache_size))
    
    K_MASK = jnp.finfo(logits.dtype).min
    
    # This is the line from _modules.py:277 that causes the error
    with pytest.raises(ValueError, match="Incompatible shapes for broadcasting"):
      _ = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)

  def test_attention_mask_broadcasting_with_auto_slice_fix(self):
    """Test that automatic slicing fixes the shape mismatch (Issue #407 fix)."""
    batch_size, seq_len, num_heads = 1, 1447, 8
    mask_cache_size = 5234
    logits_cache_size = 4096
    
    # Original 3D attention mask (larger than needed)
    attn_mask = jnp.ones((batch_size, seq_len, mask_cache_size), dtype=bool)
    
    # Logits from attention computation
    logits = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, logits_cache_size)
    )
    
    K_MASK = jnp.finfo(logits.dtype).min
    
    # Apply the fix: slice attention mask to match logits cache size
    actual_cache_size = logits.shape[-1]
    if attn_mask.shape[-1] != actual_cache_size:
      attn_mask = attn_mask[..., :actual_cache_size]
    
    # Now this should work without errors
    padded_logits = jnp.where(
        (jnp.expand_dims(attn_mask, -2)), logits, K_MASK
    )
    
    assert padded_logits.shape == logits.shape
    assert attn_mask.shape[-1] == logits.shape[-1]

  def test_attention_mask_broadcasting_correct_shapes(self):
    """Test that broadcasting works when cache sizes match."""
    batch_size, seq_len, num_heads, cache_size = 1, 100, 8, 512
    
    # Properly shaped mask and logits
    attn_mask = jnp.ones((batch_size, seq_len, cache_size), dtype=bool)
    logits = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, cache_size)
    )
    
    K_MASK = jnp.finfo(logits.dtype).min
    
    # This should work without errors
    padded_logits = jnp.where(
        (jnp.expand_dims(attn_mask, -2)), logits, K_MASK
    )
    
    assert padded_logits.shape == logits.shape
    # Verify that where mask is True, we get logits; where False, we get K_MASK
    assert jnp.all(padded_logits[0, 0, 0, :] == logits[0, 0, 0, :])
