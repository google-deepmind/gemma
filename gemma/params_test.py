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

"""Tests for transformer params."""

import os
import pathlib

from gemma import params as params_lib
import jax
import jax.numpy as jnp
import numpy as np


def _mock_params():
  """Create randomly initialized params for a Gemma transformer."""
  num_embed = 1024
  embed_dim = 512
  hidden_dim = 4 * embed_dim
  num_heads = 8
  head_dim = 256
  num_kv_heads = 4

  param_shapes = {
      'transformer': {
          'embedder': {'input_embedding': [num_embed, embed_dim]},
          'final_norm': {'scale': [embed_dim]},
          'layer_0': {
              'attn': {
                  'attn_vec_einsum': {'w': [num_heads, head_dim, embed_dim]},
                  'kv_einsum': {'w': [2, num_kv_heads, embed_dim, head_dim]},
                  'q_einsum': {'w': [num_heads, embed_dim, head_dim]},
              },
              'mlp': {
                  'gating_einsum': [2, embed_dim, hidden_dim],
                  'linear': [hidden_dim, embed_dim],
              },
              'post_attention_norm': {'scale': [embed_dim]},
              'post_ffw_norm': {'scale': [embed_dim]},
              'pre_attention_norm': {'scale': [embed_dim]},
              'pre_ffw_norm': {'scale': [embed_dim]},
          },
      }
  }

  return jax.tree_util.tree_map(
      lambda shape: jnp.zeros(shape, dtype=jnp.bfloat16), param_shapes
  )


def test_save_params(tmp_path: pathlib.Path):
  params = _mock_params()

  # Create a temporary empty directory for this unit test
  temp_dir = os.fspath(tmp_path)

  params_lib.format_and_save_params(params, temp_dir + '/checkpoint')
  params_loaded = params_lib.load_and_format_params(temp_dir + '/checkpoint')

  # Compare original with round-tripped params
  jax.tree_util.tree_map(np.testing.assert_array_equal, params, params_loaded)
