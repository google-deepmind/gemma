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

"""Regression tests for selected-canvas SFT decoding."""

from absl.testing import absltest
from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model
import jax.numpy as jnp
import numpy as np


class SelectedCanvasDecodeTest(absltest.TestCase):

  def _decode(self, xt, time):
    captured = {}

    def fake_network(*, xt, time, conditioning, is_training):
      captured['xt'] = xt
      captured['time'] = time
      captured['conditioning'] = conditioning
      captured['is_training'] = is_training
      return {'logits': xt.astype(jnp.float32)}

    prompt_len = 2
    total_canvas_len = 6
    canvas_size = 2
    positions = jnp.arange(prompt_len + total_canvas_len)[None, :]

    output = sft_model.sft_decode(
        gemma_network=fake_network,
        xt=xt,
        time=time,
        kv_cache={'sentinel': jnp.array(1)},
        positions=positions,
        prompt_mask=jnp.array([[True, True]]),
        canvas_mask=jnp.ones((1, total_canvas_len), dtype=jnp.bool_),
        selected_canvas_idx=jnp.array([1], dtype=jnp.int32),
        prompt_len=prompt_len,
        total_canvas_len=total_canvas_len,
        canvas_size=canvas_size,
        is_training=False,
    )
    return output, captured

  def test_full_response_decodes_only_selected_canvas(self):
    xt = jnp.array([[[10], [11], [20], [21], [30], [31]]])
    time = jnp.arange(6, dtype=jnp.float32)[None, :, None]

    output, captured = self._decode(xt, time)

    np.testing.assert_array_equal(captured['xt'], xt[:, 2:4])
    np.testing.assert_array_equal(captured['time'], time[:, 2:4])
    np.testing.assert_array_equal(
        captured['conditioning']['positions'], jnp.array([[4, 5]])
    )
    self.assertEqual(
        captured['conditioning']['attention_mask'].shape, (1, 2, 8)
    )
    self.assertFalse(captured['is_training'])

    self.assertEqual(output['logits'].shape, (1, 6, 1))
    np.testing.assert_array_equal(output['logits'][:, 2:4], xt[:, 2:4])
    np.testing.assert_array_equal(
        output['logits'][:, :2], jnp.zeros_like(output['logits'][:, :2])
    )
    np.testing.assert_array_equal(
        output['logits'][:, 4:], jnp.zeros_like(output['logits'][:, 4:])
    )

  def test_future_canvas_does_not_enter_decoder_queries(self):
    xt = jnp.array([[[10], [11], [20], [21], [30], [31]]])
    time = jnp.arange(6, dtype=jnp.float32)[None, :, None]
    changed_future = xt.at[:, 4:].set(99)

    output_a, captured_a = self._decode(xt, time)
    output_b, captured_b = self._decode(changed_future, time)

    np.testing.assert_array_equal(captured_a['xt'], captured_b['xt'])
    np.testing.assert_array_equal(
        output_a['logits'][:, 2:4], output_b['logits'][:, 2:4]
    )

  def test_preselected_input_is_returned_without_full_length_scatter(self):
    xt = jnp.array([[[20], [21]]])
    time = jnp.array([[[2.0], [3.0]]])

    output, captured = self._decode(xt, time)

    np.testing.assert_array_equal(captured['xt'], xt)
    self.assertEqual(output['logits'].shape, (1, 2, 1))
    np.testing.assert_array_equal(output['logits'], xt.astype(jnp.float32))


if __name__ == '__main__':
  absltest.main()
