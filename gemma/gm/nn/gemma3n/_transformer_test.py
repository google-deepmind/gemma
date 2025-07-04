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

from typing import Any

from gemma.gm.nn.gemma3n import _gemma3n as gemma3n_models
from gemma.gm.nn.gemma3n import _transformer as gt
import jax
import jax.numpy as jnp
import pytest

BATCH_SIZE = 4
SEQ_LEN = 16
NUM_IMAGES = 1


def _get_output(
    model: gt.Gemma3nTransformer, **kwargs
) -> tuple[gt.Output, Any]:

  def init_fn(**kwargs):
    out, params = model.init_with_output(jax.random.key(0), **kwargs)
    return out, params['params']

  return jax.eval_shape(init_fn, **kwargs)


@pytest.mark.parametrize(
    'model_cls',
    [
        gemma3n_models.Gemma3n_E4B,
        gemma3n_models.Gemma3n_E2B,
    ],
)
def test_transformer(model_cls: type[gt.Gemma3nTransformer]):
  model = model_cls()  # pylint: disable=missing-kwoa  # pytype: disable=missing-parameter
  tokens = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
  out, _ = _get_output(model, tokens=tokens)
  assert out.logits.shape == (BATCH_SIZE, SEQ_LEN, model.config.num_embed)


def test_images():

  model = gemma3n_models.Gemma3n_E4B()  # pylint: disable=missing-kwoa  # pytype: disable=missing-parameter

  tokens = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
  images = jnp.ones((BATCH_SIZE, NUM_IMAGES, 64, 64, 3), dtype=jnp.uint8)
  out, _ = _get_output(model, tokens=tokens, images=images)

  assert out.logits.shape == (BATCH_SIZE, SEQ_LEN, model.config.num_embed)


def test_text_only():

  model = gemma3n_models.Gemma3n_E4B(text_only=True)

  tokens = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
  images = jnp.ones((BATCH_SIZE, NUM_IMAGES, 64, 64, 3), dtype=jnp.uint8)

  with pytest.raises(ValueError, match='does not have vision encoder'):
    _get_output(model, tokens=tokens, images=images)

  out, params = _get_output(model, tokens=tokens)
  assert 'vision_encoder' not in params  # Vision params not loaded
  assert out.logits.shape == (BATCH_SIZE, SEQ_LEN, model.config.num_embed)


def test_last_only():
  model = gemma3n_models.Gemma3n_E4B(return_last_only=True)
  tokens = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
  out, params = _get_output(model, tokens=tokens)
  assert 'vision_encoder' in params  # Vision by default
  assert out.logits.shape == (BATCH_SIZE, model.config.num_embed)
