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

from unittest import mock

from gemma import gm
from gemma.gm.text import _gemma4_sampler
from gemma.gm.text import _sampler
import jax.numpy as jnp


class _Gemma4DummyTokenizer(gm.testing.DummyTokenizer):
  VERSION = 4


def _fake_state(max_out_length: int):
  state = mock.MagicMock()
  state.predicted_tokens = jnp.zeros((1, max_out_length), dtype=jnp.int32)
  state.cache_info.is_full.item.return_value = False
  return state


def test_sampler_preserves_zero_max_new_tokens():
  sampler = gm.text.Sampler(
      model=gm.testing.DummyGemma(),
      params={},
      tokenizer=gm.testing.DummyTokenizer(),
      cache_length=8,
      max_out_length=4,
      pad_length=None,
  )
  sampler_loop = mock.MagicMock()
  sampler_loop.sample.return_value = _fake_state(sampler.max_out_length)

  with mock.patch.object(_sampler._prefill, 'prefill', return_value=mock.MagicMock()):
    with mock.patch.object(
        _sampler.Sampler,
        '_initialize_sampler_loop',
        return_value=sampler_loop,
    ):
      sampler.sample('Hello world', max_new_tokens=0)

  assert int(sampler_loop.sample.call_args.kwargs['max_new_tokens']) == 0


def test_gemma4_sampler_preserves_zero_max_new_tokens():
  sampler = gm.text.Gemma4Sampler(
      model=gm.nn.Gemma4_E2B(),
      params={},
      tokenizer=_Gemma4DummyTokenizer(),
      cache_length=8,
      max_out_length=4,
      pad_length=None,
  )
  state = _fake_state(sampler.max_out_length)

  with mock.patch.object(
      _gemma4_sampler._prefill, 'prefill', return_value=mock.MagicMock()
  ):
    with mock.patch.object(
        _gemma4_sampler._sampler_loop.SamplerLoop,
        'sample',
        return_value=state,
    ) as sample_mock:
      sampler.sample('Hello world', max_new_tokens=0)

  assert int(sample_mock.call_args.kwargs['max_new_tokens']) == 0
