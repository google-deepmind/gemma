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

from etils import etree
from etils.enp.typing import f32
from flax import linen as nn
from gemma import peft
import jax
import jax.numpy as jnp


def _dense_to_lora(module):
  if isinstance(module, nn.Dense):
    return peft.LoRADense(rank=1, wrapped=module)
  elif isinstance(module, nn.Einsum):
    return peft.LoRAEinsum(rank=1, wrapped=module)
  elif isinstance(module, nn.DenseGeneral):
    return peft.LoRADenseGeneral(rank=1, wrapped=module)
  else:
    return module


class MyModule(nn.Module):

  @nn.compact
  def __call__(self, x):
    # Call a model twice
    model = nn.Dense(2)
    y0 = model(x)
    y1 = model(x)

    # Call another model
    y2 = nn.Dense(3)(x)

    # Test Einsum
    y3 = nn.Einsum(
        shape=(4, 2, 3),
        einsum_str='bi,imn->bmn',
    )(x)

    # Test DenseGeneral
    y4 = nn.DenseGeneral(
        features=(2, 3),
        axis=-1,
    )(x)

    return {
        'y0': y0,
        'y1': y1,
        'y2': y2,
        'y3': y3,
        'y4': y4,
    }


def test_lora():
  model = MyModule()
  with peft.ModuleInterceptor(_dense_to_lora):
    out, params = model.init_with_output(
        jax.random.key(0),
        jnp.zeros((1, 4)),
    )

  assert etree.spec_like(params) == etree.spec_like({
      'params': {
          'Dense_0': {
              'kernel': f32[4, 2],
              'bias': f32[2],
              'lora': {
                  'a': f32[4, 1],
                  'b': f32[1, 2],
              },
          },
          'Dense_1': {
              'kernel': f32[4, 3],
              'bias': f32[3],
              'lora': {
                  'a': f32[4, 1],
                  'b': f32[1, 3],
              },
          },
          'Einsum_0': {
              'kernel': f32[4, 2, 3],
              'bias': f32[2, 3],
              'lora': {
                  'a': f32[4, 1],
                  'b': f32[1, 2, 3],
              },
          },
          'DenseGeneral_0': {
              'kernel': f32[4, 2, 3],
              'bias': f32[2, 3],
              'lora': {
                  'a': f32[4, 1],
                  'b': f32[1, 2, 3],
              },
          },
      },
  })
  assert etree.spec_like(out) == etree.spec_like({
      'y0': f32[1, 2],
      'y1': f32[1, 2],
      'y2': f32[1, 3],
      'y3': f32[1, 2, 3],
      'y4': f32[1, 2, 3],
  })
