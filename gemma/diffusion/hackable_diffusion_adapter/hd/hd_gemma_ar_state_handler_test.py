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

"""Unit tests for GemmaARStateHandler attention mask creation."""

from unittest import mock

from absl.testing import absltest
import flax.linen as nn
from gemma.diffusion import _models
from gemma.diffusion import _transformer as diffusion_transformer
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_ar_state_handler
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network
from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
from gemma.gm.nn.gemma4 import _config
from gemma.gm.nn.gemma4 import _modules
import jax
import jax.numpy as jnp
import numpy as np

################################################################################
# MARK: Helpers
################################################################################


def _make_small_config():
  """Create a tiny TransformerConfig for testing."""
  return _config.TransformerConfig(
      num_embed=32,
      embed_dim=8,
      num_heads=2,
      num_kv_heads=1,
      head_dim=4,
      hidden_dim=16,
      attention_types=[_modules.AttentionType.GLOBAL],
      kv_cache_sharing_config=None,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      final_logit_softcap=None,
      global_rope_proportion=1.0,
  )


def _make_handler(cache_length=16):
  """Create a GemmaARStateHandler with a tiny Gemma model."""
  small_config = _make_small_config()
  gemma_model = _models.DiffusionGemma_A26B_A4B(
      config=small_config,
      self_conditioning_config=diffusion_transformer.SelfConditioningConfig(
          features=small_config.embed_dim,
          hidden_dim=small_config.hidden_dim,
      ),
  )

  # We need to init the params through a Flax module so that the model is
  # properly bound.  We create a thin wrapper that inits the cache and runs
  # a prefill pass to materialise all parameters.
  gemma_network = hd_gemma_network.WrappedDiffusionGemmaNetwork(
      gemma_model=gemma_model,
  )

  class _InitWrapper(nn.Module):
    net: hd_gemma_network.WrappedDiffusionGemmaNetwork

    @nn.compact
    def __call__(self, tokens, input_mask, cache_length):
      cache = self.net.init_cache(
          batch_size=tokens.shape[0], cache_length=cache_length
      )
      positions = mask_helpers.build_positions_from_mask(input_mask)
      attention_mask = mask_helpers.make_causal_prefill_mask(
          input_mask, cache_length
      )
      out = self.net.encoder_call(
          x=tokens,
          conditioning_embeddings={
              "kv_cache": cache,
              "positions": positions,
              "attention_mask": attention_mask,
          },
      )
      return out

  wrapper = _InitWrapper(net=gemma_network)
  # Use a small prompt to init.
  dummy_tokens = jnp.ones((1, 4), dtype=jnp.int32)
  dummy_mask = jnp.ones((1, 4), dtype=jnp.bool_)
  rng = jax.random.PRNGKey(0)
  variables = wrapper.init(rng, dummy_tokens, dummy_mask, cache_length)
  params = variables["params"]["net"]["gemma_model"]

  handler = hd_gemma_ar_state_handler.GemmaARStateHandler(
      gemma_network=gemma_network,
      gemma_params=params,
      end_tokens=(2,),
      pad_token=0,
  )
  return handler


def _make_conditioning(prompt_tokens, prompt_lengths):
  """Build a conditioning dict from prompt tokens and lengths."""
  return {
      "prompt_tokens": jnp.array(prompt_tokens, dtype=jnp.int32),
      "prompt_lengths": jnp.array(prompt_lengths, dtype=jnp.int32),
  }


def _make_dummy_canvas_last_step(canvas_tokens):
  """Build a mock DiffusionStepTree with the given canvas tokens."""
  # canvas_last_step.xt shape: (B, L, 1)
  xt = jnp.array(canvas_tokens, dtype=jnp.int32)[..., None]
  step_info = mock.MagicMock()
  step_info.step = 10
  return mock.MagicMock(
      xt=xt,
      step_info=step_info,
  )


################################################################################
# MARK: init_ar_state
################################################################################


class InitARStateFullAttentionMaskTest(absltest.TestCase):
  """Tests for the full_attention_mask created by init_ar_state."""

  def test_full_attention_mask_hides_padding(self):
    """full_attention_mask should be False at prompt pad positions.

    prompt = [5, 10, 0, 0, 0]  (lengths=[2])
    full_attention_mask should be:
      [True, True, False, False, False, True, True, ..., True]
       real   real   pad    pad    pad   decode slots
    """
    cache_length = 13  # 5 + 2 * 4
    handler = _make_handler(cache_length=cache_length)
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 0, 0, 0]],
        prompt_lengths=[2],
    )
    state = handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=4,
        max_num_canvases=2,
    )
    full_mask = state["full_attention_mask"]  # (B, cache_length)

    # Prompt region: positions 0,1 are True; 2,3,4 are False (pad).
    np.testing.assert_array_equal(full_mask[0, :2], [True, True])
    np.testing.assert_array_equal(full_mask[0, 2:5], [False, False, False])

    # All decode slots (positions 5 onwards) should be True.
    np.testing.assert_array_equal(
        full_mask[0, 5:], np.ones(cache_length - 5, dtype=bool)
    )

  def test_full_attention_mask_no_padding(self):
    """When there is no padding, the entire mask should be True."""
    cache_length = 11  # 3 + 2 * 4
    handler = _make_handler(cache_length=cache_length)
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 15]],
        prompt_lengths=[3],
    )
    state = handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=4,
        max_num_canvases=2,
    )
    full_mask = state["full_attention_mask"]
    np.testing.assert_array_equal(
        full_mask[0], np.ones(cache_length, dtype=bool)
    )


class InitARStateCanvasAttnMaskTest(absltest.TestCase):
  """Tests for canvas_attn_mask created by init_ar_state."""

  def setUp(self):
    super().setUp()
    self.cache_length = 13  # 5 + 2 * 4

  def test_canvas_attn_mask_hides_padding_columns(self):
    """Canvas attention mask should never attend to prompt pad KV slots.

    prompt = [5, 10, 0, 0, 0]  (lengths=[2]), canvas_length=4
    Padding is at prompt positions 2,3,4 → those columns should be False
    for ALL canvas query positions.
    """
    handler = _make_handler(cache_length=self.cache_length)
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 0, 0, 0]],
        prompt_lengths=[2],
    )
    state = handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=4,
        max_num_canvases=2,
    )
    attn_mask = state["attention_mask"]  # (B, canvas_length, cache_length)

    # Pad columns (2, 3, 4) should be False for every canvas query position.
    pad_columns = attn_mask[0, :, 2:5]  # (canvas_length, 3)
    np.testing.assert_array_equal(
        pad_columns, np.zeros_like(pad_columns, dtype=bool)
    )

  def test_canvas_attn_mask_attends_to_real_prompt(self):
    """All canvas tokens should attend to real prompt tokens.

    prompt = [5, 10, 0, 0, 0]  (lengths=[2]), canvas_length=4
    Columns 0 and 1 (real prompt) should be True for all canvas queries.
    """
    handler = _make_handler(cache_length=self.cache_length)
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 0, 0, 0]],
        prompt_lengths=[2],
    )
    state = handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=4,
        max_num_canvases=2,
    )
    attn_mask = state["attention_mask"]

    # Real prompt columns (0, 1) should be True for all canvas queries.
    prompt_columns = attn_mask[0, :, :2]  # (canvas_length, 2)
    np.testing.assert_array_equal(
        prompt_columns, np.ones_like(prompt_columns, dtype=bool)
    )

  def test_canvas_attn_mask_hides_future_cache_slots(self):
    """Cache slots beyond (end_index + canvas_length) should be False.

    After prefill, end_index = max_prompt_len = 5.
    Canvas length = 4, so used_after_canvas = 9.
    Cache slots [9, cache_length) should be False.
    """
    handler = _make_handler(cache_length=self.cache_length)
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 0, 0, 0]],
        prompt_lengths=[2],
    )
    state = handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=4,
        max_num_canvases=2,
    )
    attn_mask = state["attention_mask"]
    # end_index = max_prompt_len = 5, canvas_length = 4
    # used_after_canvas = 5 + 4 = 9
    # So cache slots 9..15 should be masked out.
    future_columns = attn_mask[0, :, 9:]  # (canvas_length, 7)
    np.testing.assert_array_equal(
        future_columns, np.zeros_like(future_columns, dtype=bool)
    )

  def test_canvas_attn_mask_reveals_first_canvas_window(self):
    """Canvas should attend to the decode slots where it will be written.

    With prompt len 5, end_index = 5, canvas_length = 4.
    Decode slots 5,6,7,8 are the first canvas window and should be attended.
    """
    handler = _make_handler(cache_length=self.cache_length)
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 15, 0, 0]],
        prompt_lengths=[3],
    )
    state = handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=4,
        max_num_canvases=2,
    )
    attn_mask = state["attention_mask"]

    # Decode slots 5,6,7,8 should be True for all canvas positions.
    canvas_window = attn_mask[0, :, 5:9]  # (canvas_length=4, 4)
    np.testing.assert_array_equal(
        canvas_window, np.ones_like(canvas_window, dtype=bool)
    )

  def test_canvas_attn_mask_shape(self):
    """Canvas attention mask should have shape (B, canvas_length, cache_length)."""
    handler = _make_handler(cache_length=self.cache_length)
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 0, 0, 0]],
        prompt_lengths=[2],
    )
    canvas_length = 4
    state = handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=canvas_length,
        max_num_canvases=2,
    )
    attn_mask = state["attention_mask"]
    self.assertEqual(attn_mask.shape, (1, canvas_length, self.cache_length))


class InitARStatePositionsTest(absltest.TestCase):
  """Tests for canvas positions created by init_ar_state."""

  def test_positions_start_after_real_prompt(self):
    """Canvas positions should start at prompt_length, not max_prompt_len.

    prompt = [5, 10, 0, 0, 0]  (lengths=[2]), canvas_length=4
    Positions should be [2, 3, 4, 5], not [5, 6, 7, 8].
    """
    handler = _make_handler(cache_length=13)  # 5 + 2 * 4
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 0, 0, 0]],
        prompt_lengths=[2],
    )
    state = handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=4,
        max_num_canvases=2,
    )
    positions = state["positions"]  # (B, canvas_length)
    np.testing.assert_array_equal(positions[0], [2, 3, 4, 5])

  def test_positions_batched_different_lengths(self):
    """Each batch element should have positions starting after its own prompt.

    prompt = [[5, 10, 15, 0],   (lengths=[3])
              [5, 10, 0, 0]]    (lengths=[2])
    canvas_length = 3
    Element 0 positions: [3, 4, 5]
    Element 1 positions: [2, 3, 4]
    """
    handler = _make_handler(cache_length=10)  # 4 + 2 * 3
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 15, 0], [5, 10, 0, 0]],
        prompt_lengths=[3, 2],
    )
    state = handler.init_ar_state(
        batch_size=2,
        conditioning=conditioning,
        canvas_length=3,
        max_num_canvases=2,
    )
    positions = state["positions"]
    np.testing.assert_array_equal(positions[0], [3, 4, 5])
    np.testing.assert_array_equal(positions[1], [2, 3, 4])


class InitARStatePaddingInvarianceTest(absltest.TestCase):
  """Tests that masks are invariant to padding amount."""

  def test_full_attention_mask_same_pattern_different_padding(self):
    """The real-token portion of full_attention_mask should not change.

    prompt_1 = [5, 10, 0, 0]     (lengths=[2], max_prompt_len=4)
    prompt_2 = [5, 10, 0, 0, 0]  (lengths=[2], max_prompt_len=5)

    The first 2 positions should be True, then pad positions False,
    then decode slots True — the pattern at real tokens is the same.
    """
    handler_1 = _make_handler(cache_length=12)  # 4 + 2 * 4
    handler_2 = _make_handler(cache_length=13)  # 5 + 2 * 4

    cond_1 = _make_conditioning(
        prompt_tokens=[[5, 10, 0, 0]],
        prompt_lengths=[2],
    )
    cond_2 = _make_conditioning(
        prompt_tokens=[[5, 10, 0, 0, 0]],
        prompt_lengths=[2],
    )
    state_1 = handler_1.init_ar_state(
        batch_size=1, conditioning=cond_1, canvas_length=4, max_num_canvases=2
    )
    state_2 = handler_2.init_ar_state(
        batch_size=1, conditioning=cond_2, canvas_length=4, max_num_canvases=2
    )
    fm_1 = state_1["full_attention_mask"][0]
    fm_2 = state_2["full_attention_mask"][0]

    # Real prompt tokens (first 2) are True in both.
    np.testing.assert_array_equal(fm_1[:2], [True, True])
    np.testing.assert_array_equal(fm_2[:2], [True, True])

    # Pad region in both should be False.
    np.testing.assert_array_equal(fm_1[2:4], [False, False])
    np.testing.assert_array_equal(fm_2[2:5], [False, False, False])


################################################################################
# Tests: update_ar_state
################################################################################


class UpdateARStateAttentionMaskTest(absltest.TestCase):
  """Tests for attention mask updates in update_ar_state."""

  def setUp(self):
    super().setUp()
    self.cache_length = 17  # 5 + 3 * 4 (also works for 5 + 4 * 3)
    self.handler = _make_handler(cache_length=self.cache_length)

  def test_attention_mask_reveals_next_canvas_window(self):
    """After update, the attention mask should reveal the next canvas window.

    Setup: prompt=[5,10,15,0,0] (len=3), canvas_length=4, cache_length=24.
    After init: end_index=5, used_after_canvas=9.
    After first update (append 4 tokens): end_index=9, used_after_next=13.
    The mask should reveal cache slots [0..12] minus pad slots [3,4].
    """
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 15, 0, 0]],
        prompt_lengths=[3],
    )
    canvas_length = 4
    state = self.handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=canvas_length,
        max_num_canvases=3,
    )

    # Simulate a canvas with no stop tokens.
    canvas = _make_dummy_canvas_last_step(
        canvas_tokens=[[7, 8, 9, 11]],
    )
    state = self.handler.update_ar_state(
        canvas_last_step=canvas, sampler_state=state
    )

    attn_mask = state["attention_mask"]  # (B, canvas_length, cache_length)

    # Real prompt columns (0,1,2) should be True.
    np.testing.assert_array_equal(
        attn_mask[0, :, :3], np.ones((canvas_length, 3), dtype=bool)
    )

    # Pad columns (3,4) should be False.
    np.testing.assert_array_equal(
        attn_mask[0, :, 3:5], np.zeros((canvas_length, 2), dtype=bool)
    )

    # First canvas window (5..8) should be True (already written).
    np.testing.assert_array_equal(
        attn_mask[0, :, 5:9], np.ones((canvas_length, 4), dtype=bool)
    )

    # Second canvas window (9..12) should be True (being written next).
    np.testing.assert_array_equal(
        attn_mask[0, :, 9:13], np.ones((canvas_length, 4), dtype=bool)
    )

    # Future slots (13 onwards) should be False.
    np.testing.assert_array_equal(
        attn_mask[0, :, 13:],
        np.zeros((canvas_length, self.cache_length - 13), dtype=bool),
    )

  def test_attention_mask_still_hides_padding_after_update(self):
    """Prompt padding should stay hidden even after multiple updates."""
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 0, 0, 0]],
        prompt_lengths=[2],
    )
    canvas_length = 3
    state = self.handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=canvas_length,
        max_num_canvases=4,
    )

    # Run two updates.
    for token_offset in range(2):
      canvas = _make_dummy_canvas_last_step(
          canvas_tokens=[[7 + token_offset, 8, 9]],
      )
      state = self.handler.update_ar_state(
          canvas_last_step=canvas, sampler_state=state
      )

    attn_mask = state["attention_mask"]

    # Pad columns 2,3,4 should remain False after all updates.
    np.testing.assert_array_equal(
        attn_mask[0, :, 2:5],
        np.zeros((canvas_length, 3), dtype=bool),
    )


class UpdateARStatePositionsTest(absltest.TestCase):
  """Tests for position updates in update_ar_state."""

  def test_positions_advance_by_canvas_length(self):
    """After update, positions should advance by exactly canvas_length."""
    handler = _make_handler(cache_length=17)  # 5 + 3 * 4
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 15, 0, 0]],
        prompt_lengths=[3],
    )
    canvas_length = 4
    state = handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=canvas_length,
        max_num_canvases=3,
    )

    initial_positions = np.array(state["positions"])

    canvas = _make_dummy_canvas_last_step(
        canvas_tokens=[[7, 8, 9, 11]],
    )
    state = handler.update_ar_state(
        canvas_last_step=canvas, sampler_state=state
    )

    np.testing.assert_array_equal(
        state["positions"], initial_positions + canvas_length
    )


class UpdateARStatePredictedTokensTest(absltest.TestCase):
  """Tests for predicted_tokens buffer updates."""

  def test_canvas_written_to_correct_buffer_position(self):
    """Canvas tokens should be written at the correct offset in the buffer."""
    handler = _make_handler(cache_length=17)  # 5 + 3 * 4
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 15, 0, 0]],
        prompt_lengths=[3],
    )
    canvas_length = 4
    state = handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=canvas_length,
        max_num_canvases=3,
    )

    prompt_len = 5  # max_prompt_len
    canvas_tokens = [[7, 8, 9, 11]]
    canvas = _make_dummy_canvas_last_step(
        canvas_tokens=canvas_tokens,
    )
    state = handler.update_ar_state(
        canvas_last_step=canvas, sampler_state=state
    )

    predicted = state["predicted_tokens"]
    # Canvas should be written at positions [prompt_len, prompt_len+4).
    np.testing.assert_array_equal(
        predicted[0, prompt_len : prompt_len + canvas_length],
        [7, 8, 9, 11],
    )


class UpdateARStateStopTokenTest(absltest.TestCase):
  """Tests for stop token handling in update_ar_state."""

  def test_done_flag_set_on_stop_token(self):
    """When a canvas contains a stop token, done should be set to True."""
    handler = _make_handler(cache_length=17)  # 5 + 3 * 4
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 15, 0, 0]],
        prompt_lengths=[3],
    )
    canvas_length = 4
    state = handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=canvas_length,
        max_num_canvases=3,
    )

    # end_tokens=(2,), so token 2 is a stop token.
    canvas = _make_dummy_canvas_last_step(
        canvas_tokens=[[7, 2, 9, 11]],
    )
    state = handler.update_ar_state(
        canvas_last_step=canvas, sampler_state=state
    )

    self.assertTrue(bool(state["done"][0]))

  def test_tokens_after_stop_are_padded(self):
    """Tokens after a stop token should be replaced with pad_token."""
    handler = _make_handler(cache_length=17)  # 5 + 3 * 4
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 15, 0, 0]],
        prompt_lengths=[3],
    )
    canvas_length = 4
    state = handler.init_ar_state(
        batch_size=1,
        conditioning=conditioning,
        canvas_length=canvas_length,
        max_num_canvases=3,
    )

    # Stop token at position 1 (token=2), so positions 2,3 should be padded.
    canvas = _make_dummy_canvas_last_step(
        canvas_tokens=[[7, 2, 9, 11]],
    )
    state = handler.update_ar_state(
        canvas_last_step=canvas, sampler_state=state
    )

    prompt_len = 5  # max_prompt_len
    written = state["predicted_tokens"][0, prompt_len : prompt_len + 4]
    # Token 7 kept, stop token 2 kept, positions 2,3 replaced with pad (0).
    np.testing.assert_array_equal(written, [7, 2, 0, 0])


################################################################################
# MARK: Batched behaviour
################################################################################


class BatchedMaskTest(absltest.TestCase):
  """Tests with batched inputs with different prompt lengths."""

  def test_per_element_padding_masks(self):
    """Each batch element should have its own padding pattern in masks.

    Element 0: prompt=[5, 10, 15, 0] (len=3) → pad at col 3
    Element 1: prompt=[5, 0, 0, 0]   (len=1) → pad at cols 1,2,3
    """
    handler = _make_handler(cache_length=12)  # 4 + 2 * 4
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 15, 0], [5, 0, 0, 0]],
        prompt_lengths=[3, 1],
    )
    state = handler.init_ar_state(
        batch_size=2,
        conditioning=conditioning,
        canvas_length=4,
        max_num_canvases=2,
    )
    full_mask = state["full_attention_mask"]  # (B=2, cache_length=12)

    # Element 0: positions 0,1,2 True; position 3 False.
    np.testing.assert_array_equal(full_mask[0, :3], [True, True, True])
    self.assertFalse(bool(full_mask[0, 3]))

    # Element 1: position 0 True; positions 1,2,3 False.
    self.assertTrue(bool(full_mask[1, 0]))
    np.testing.assert_array_equal(full_mask[1, 1:4], [False, False, False])

  def test_per_element_positions(self):
    """Each batch element should have positions starting at its own length."""
    handler = _make_handler(cache_length=12)  # 4 + 2 * 4
    conditioning = _make_conditioning(
        prompt_tokens=[[5, 10, 15, 0], [5, 0, 0, 0]],
        prompt_lengths=[3, 1],
    )
    state = handler.init_ar_state(
        batch_size=2,
        conditioning=conditioning,
        canvas_length=4,
        max_num_canvases=2,
    )
    positions = state["positions"]
    np.testing.assert_array_equal(positions[0], [3, 4, 5, 6])
    np.testing.assert_array_equal(positions[1], [1, 2, 3, 4])


################################################################################
# MARK: truncate_canvas_at_stop_tokens (auxiliary function)
################################################################################


class TruncateCanvasTest(absltest.TestCase):
  """Tests for truncate_canvas_at_stop_tokens."""

  def test_no_stop_token(self):
    """Canvas without stop tokens should be returned unchanged."""
    canvas = jnp.array([[7, 8, 9, 11]], dtype=jnp.int32)
    done = jnp.array([False])
    result, has_stop = hd_gemma_ar_state_handler.truncate_canvas_at_stop_tokens(
        canvas=canvas,
        end_tokens=(2,),
        canvas_length=4,
        done=done,
    )
    np.testing.assert_array_equal(result, [[7, 8, 9, 11]])
    self.assertFalse(bool(has_stop[0]))

  def test_stop_token_at_start(self):
    """Stop token at position 0 should keep only that token."""
    canvas = jnp.array([[2, 8, 9, 11]], dtype=jnp.int32)
    done = jnp.array([False])
    result, has_stop = hd_gemma_ar_state_handler.truncate_canvas_at_stop_tokens(
        canvas=canvas,
        end_tokens=(2,),
        canvas_length=4,
        done=done,
    )
    np.testing.assert_array_equal(result, [[2, 0, 0, 0]])
    self.assertTrue(bool(has_stop[0]))

  def test_already_done_is_fully_padded(self):
    """If the batch element is already done, the entire canvas is padded."""
    canvas = jnp.array([[7, 8, 9, 11]], dtype=jnp.int32)
    done = jnp.array([True])
    result, has_stop = hd_gemma_ar_state_handler.truncate_canvas_at_stop_tokens(
        canvas=canvas,
        end_tokens=(2,),
        canvas_length=4,
        done=done,
    )
    np.testing.assert_array_equal(result, [[0, 0, 0, 0]])
    self.assertFalse(bool(has_stop[0]))


################################################################################
# MARK: Miscellaneous tests
################################################################################


class PropagateSelfConditioningFnTest(absltest.TestCase):

  def test_call(self):
    fn = hd_gemma_ar_state_handler.PropagateSelfConditioningFn()
    conditioning = {"other": 1}
    step_carry = mock.MagicMock()
    step_carry.aux = {"logits": jnp.array([1, 2])}

    res = fn(conditioning, step_carry)
    self.assertEqual(res["sc_logits"].tolist(), [1, 2])
    self.assertEqual(res["other"], 1)


class GemmaARStateHandlerTest(absltest.TestCase):

  def test_update_from_context(self):
    handler = hd_gemma_ar_state_handler.GemmaARStateHandler(
        gemma_network=mock.MagicMock(),
        gemma_params=None,
        end_tokens=(1,),
        pad_token=0,
    )

    # Create a mock Kauldron context
    context = mock.MagicMock()

    # Mock get_by_path to return the params
    with mock.patch("kauldron.kontext.get_by_path") as mock_get:
      mock_get.return_value = {"some_params": 1}
      handler.update_from_context(context)

      self.assertEqual(handler.gemma_params, {"some_params": 1})
      mock_get.assert_called_with(context.params, "gemma_network.gemma_model")


if __name__ == "__main__":
  absltest.main()
