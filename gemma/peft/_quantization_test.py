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
from etils.enp.typing import ArrayAliasMeta  # pylint: disable=g-importing-member
from etils.enp.typing import f32
from flax import linen as nn
from gemma import peft
import jax
from jax import numpy as jnp
import numpy as np

i4 = ArrayAliasMeta(shape=None, dtype=jnp.int4)


def _dense_to_quantized(module):
  if isinstance(module, nn.Dense):
    return peft.SimulateQuantizedDense(
        wrapped=module, method=peft.QuantizationMethod.INT4
    )
  if isinstance(module, nn.Einsum):
    return peft.SimulateQuantizedEinsum(
        wrapped=module, method=peft.QuantizationMethod.INT4
    )
  else:
    return module


def _dense_to_int4(module):
  if isinstance(module, nn.Dense):
    return peft.IntDense(wrapped=module, dtype=jnp.int4)
  if isinstance(module, nn.Einsum):
    return peft.IntEinsum(wrapped=module, dtype=jnp.int4)
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
    return {
        'y0': y0,
        'y1': y1,
        'y2': y2,
        'y3': y3,
    }


def test_quantization_simulation():
  model = MyModule()
  with peft.ModuleInterceptor(_dense_to_quantized):
    out, params = model.init_with_output(
        jax.random.key(0),
        jnp.zeros((1, 4)),
    )

  assert etree.spec_like(params) == etree.spec_like({
      'params': {
          'Dense_0': {
              'kernel': f32[4, 2],
              'bias': f32[2],
          },
          'Dense_1': {
              'kernel': f32[4, 3],
              'bias': f32[3],
          },
          'Einsum_0': {
              'kernel': f32[4, 2, 3],
              'bias': f32[2, 3],
          },
      },
  })
  assert etree.spec_like(out) == etree.spec_like({
      'y0': f32[1, 2],
      'y1': f32[1, 2],
      'y2': f32[1, 3],
      'y3': f32[1, 2, 3],
  })


def test_quantization():
  model = MyModule()
  params = model.init(jax.random.key(0), jnp.zeros((1, 4)))
  params_q = peft.quantize(
      params,
      method=peft.QuantizationMethod.INT4,
      in_place_keys=True,
      checkpoint_kernel_key='kernel',
  )
  with peft.ModuleInterceptor(_dense_to_int4):
    out = model.apply(params_q, jnp.zeros((1, 4)))
  assert etree.spec_like(params_q) == etree.spec_like({
      'params': {
          'Dense_0': {
              'kernel': i4[4, 2],
              'bias': f32[2],
              'scale': f32[1, 2],
          },
          'Dense_1': {
              'kernel': i4[4, 3],
              'bias': f32[3],
              'scale': f32[1, 3],
          },
          'Einsum_0': {
              'kernel': i4[4, 2, 3],
              'bias': f32[2, 3],
              'scale': f32[1, 1, 3],
          },
      },
  })
  assert etree.spec_like(out) == etree.spec_like({
      'y0': f32[1, 2],
      'y1': f32[1, 2],
      'y2': f32[1, 3],
      'y3': f32[1, 2, 3],
  })
