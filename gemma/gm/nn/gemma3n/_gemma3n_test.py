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

import dataclasses

from gemma.gm.nn.gemma3n import _gemma3n as gemma3n_models
from gemma.gm.nn.gemma3n import _transformer as gt
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    'model_cls',
    [
        gemma3n_models.Gemma3n_E4B,
        gemma3n_models.Gemma3n_E2B,
    ],
)
def test_final_logit_softcap(model_cls: type[gt.Gemma3nTransformer]):
  # Gemma 3n soft-caps its output logits: `30 * tanh(logits / 30)`
  # (`final_logit_softcapping: 30.0` in the released Gemma 3n configs).
  assert model_cls.config.final_logit_softcap == 30.0


def test_logits_are_soft_capped():
  # A small model with the Gemma 3n config, so the logits can be computed.
  config = dataclasses.replace(
      gemma3n_models._get_gemma3n_config(  # pylint: disable=protected-access
          hidden_dim=64, num_layers=5, num_unshared_kv_layers=5
      ),
      num_embed=64,
      embed_dim=32,
      num_heads=2,
      head_dim=16,
      num_kv_heads=1,
      per_layer_input_dim=8,
      laurel_rank=4,
      vision_encoder=None,
  )
  uncapped_config = dataclasses.replace(config, final_logit_softcap=None)
  model = gt.Gemma3nTransformer(config=config, dtype=jnp.float32)
  uncapped_model = gt.Gemma3nTransformer(
      config=uncapped_config, dtype=jnp.float32
  )

  tokens = jnp.arange(16, dtype=jnp.int32)[None]
  params = model.init(jax.random.key(0), tokens=tokens)
  # Random weights of a large scale, so the raw logits exceed the cap.
  leaves, treedef = jax.tree.flatten(params)
  keys = jax.random.split(jax.random.key(1), len(leaves))
  params = jax.tree.unflatten(
      treedef,
      [
          2.0 * jax.random.normal(k, x.shape, x.dtype)
          for k, x in zip(keys, leaves)
      ],
  )

  logits = model.apply(params, tokens=tokens).logits
  raw_logits = uncapped_model.apply(params, tokens=tokens).logits
  assert np.abs(raw_logits).max() > 30.0
  assert np.abs(logits).max() <= 30.0
  np.testing.assert_allclose(
      logits, 30.0 * np.tanh(np.asarray(raw_logits) / 30.0), atol=1e-4
  )
