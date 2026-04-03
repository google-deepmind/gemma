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

from typing import Any

from gemma.gm.nn.gemma4 import _gemma4 as gemma4_models
from gemma.gm.nn.gemma4 import _transformer as gt
import jax
import jax.numpy as jnp
import pytest


BATCH_SIZE = 4
SEQ_LEN = 16


def _get_output(
    model: gt.Transformer, **kwargs
) -> tuple[gt.Output, Any]:

  def init_fn(**kwargs):
    out, params = model.init_with_output(jax.random.key(0), **kwargs)
    return out, params['params']

  return jax.eval_shape(init_fn, **kwargs)


@pytest.mark.parametrize(
    'model_cls',
    [
        gemma4_models.Gemma4_E2B,
        gemma4_models.Gemma4_E4B,
        gemma4_models.Gemma4_31B,
    ],
)
def test_transformer(model_cls: type[gt.Transformer]):
  model = model_cls()  # pylint: disable=missing-kwoa  # pytype: disable=missing-parameter
  tokens = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
  out, _ = _get_output(model, tokens=tokens)
  assert out.logits.shape == (BATCH_SIZE, SEQ_LEN, model.config.num_embed)


def test_text_only():
  model = gemma4_models.Gemma4_31B(text_only=True)  # pylint: disable=missing-kwoa  # pytype: disable=missing-parameter
  tokens = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
  out, params = _get_output(model, tokens=tokens)
  assert 'vision_encoder' not in params
  assert out.logits.shape == (BATCH_SIZE, SEQ_LEN, model.config.num_embed)
