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

"""Tests for LoRA wrapper delegation."""

from absl.testing import absltest
from flax import linen as nn
from gemma.gm.nn import _lora
import jax
import jax.numpy as jnp


class DummyModel(nn.Module):
  some_attr: int = 42

  @nn.compact
  def __call__(self, x):
    return nn.Dense(features=4, name='dense')(x)

  @nn.compact
  def encoder_call(self, x):
    return nn.Dense(features=4, name='dense')(x)

  @nn.compact
  def init_cache(self, x):
    return nn.Dense(features=4, name='dense')(x)


class LoRATest(absltest.TestCase):

  def test_call_intercepts_dense(self):
    model = _lora.LoRA(rank=2, model=DummyModel())
    params = model.init(jax.random.key(0), jnp.zeros((1, 4)))['params']
    self.assertIn('dense', params)
    self.assertIn('lora', params['dense'])
    self.assertIn('a', params['dense']['lora'])
    self.assertIn('b', params['dense']['lora'])

  def test_encoder_call_intercepts_dense(self):
    model = _lora.LoRA(rank=2, model=DummyModel())
    params = model.init(
        jax.random.key(0), jnp.zeros((1, 4)), method=model.encoder_call
    )['params']
    self.assertIn('dense', params)
    self.assertIn('lora', params['dense'])
    self.assertIn('a', params['dense']['lora'])
    self.assertIn('b', params['dense']['lora'])

  def test_init_cache_intercepts_dense(self):
    model = _lora.LoRA(rank=2, model=DummyModel())
    params = model.init(
        jax.random.key(0), jnp.zeros((1, 4)), method=model.init_cache
    )['params']
    self.assertIn('dense', params)
    self.assertIn('lora', params['dense'])
    self.assertIn('a', params['dense']['lora'])
    self.assertIn('b', params['dense']['lora'])

  def test_getattr_forwarding(self):
    model = _lora.LoRA(rank=2, model=DummyModel())
    self.assertEqual(model.some_attr, 42)


if __name__ == '__main__':
  absltest.main()
