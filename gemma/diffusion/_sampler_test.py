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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dialog
from gemma.diffusion import _chat_sampler
from gemma.diffusion import _early_stopping
from gemma.diffusion import _models
from gemma.diffusion import _sampler
from gemma.diffusion import _transformer
from gemma.gm.nn.gemma4 import _config
from gemma.gm.nn.gemma4 import _modules
from gemma.gm.nn.gemma4 import _transformer as _gemma4_transformer
from gemma.gm.text import _sampler_loop
from gemma.gm.text import _sampling
from gemma.gm.text import _tokenizer
import jax
import jax.numpy as jnp
import numpy as np


class _MockSpecialTokens(_tokenizer.SpecialTokens):
  PAD = 0
  EOS = 1
  BOS = 2
  UNK = 3
  MASK = 4
  CUSTOM = 5
  START_OF_TURN = 6
  END_OF_TURN = 7
  START_OF_IMAGE = 8
  END_OF_IMAGE = 9
  BEGIN_OF_TOOL_RESPONSE = 10
  END_OF_TOOL_RESPONSE = 11


@dataclasses.dataclass(frozen=True)
class _MockTokenizer:
  vocab_size: int = 32

  VERSION = None
  FORBIDDEN_TOKENS = ()
  FORMAT = dialog.Format.GEMMA3

  special_tokens = _MockSpecialTokens

  def encode(self, text, *, add_bos=False, add_eos=False):
    tokens = [ord(c) % self.vocab_size for c in text]
    if add_bos:
      tokens.insert(0, self.special_tokens.BOS)
    if add_eos:
      tokens.append(self.special_tokens.EOS)
    return tokens

  def decode(self, ids):
    if hasattr(ids, 'tolist'):
      ids = ids.tolist()
    if isinstance(ids, int):
      ids = [ids]
    return ''.join(chr(i + 65) if 0 <= i < 26 else '?' for i in ids)


_SMALL_CONFIG = _config.TransformerConfig(
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

_SMALL_SC_CONFIG = _transformer.SelfConditioningConfig(
    features=_SMALL_CONFIG.embed_dim,
    hidden_dim=_SMALL_CONFIG.hidden_dim,
)


def _sample_with_while_loop(
    sampler,
    params,
    cache,
    batch_size,
    canvas_length,
    max_denoising_steps,
    rng,
):
  """Runs sample_next_canvas which uses jax.lax.while_loop internally."""
  return sampler.sample_next_canvas(
      canvas_length=canvas_length,
      max_denoising_steps=max_denoising_steps,
      batch_size=batch_size,
      cache=cache,
      params=params,
      rng=rng,
  )


def _sample_with_for_loop(
    sampler,
    params,
    cache,
    batch_size,
    canvas_length,
    embed_dim,
    vocab_size,
    max_denoising_steps,
    rng,
    num_steps_override=None,
):
  """Reimplements the denoising loop with a Python for loop."""
  initial_canvas_rng, step_rng = jax.random.split(rng)

  if cache is not None:
    cache_layer = list(cache.values())[0]
    cache_len = cache_layer['k'].shape[1]
    samples_in_cache = cache_layer['end_index']
    positions = samples_in_cache[:, None] + jnp.arange(canvas_length)[None, :]
  else:
    cache_len = None
    samples_in_cache = None
    positions = jnp.broadcast_to(
        jnp.arange(canvas_length)[None, :], (batch_size, canvas_length)
    )

  attention_mask = _sampler._make_global_attention_mask(
      batch_size=batch_size,
      canvas_length=canvas_length,
      cache_length=cache_len,
      num_valid_tokens=samples_in_cache,
  )

  canvas = sampler.diffusion_process.get_initial_sample(
      rng=initial_canvas_rng,
      batch_size=batch_size,
      canvas_length=canvas_length,
      text_vocab_size=vocab_size,
  )
  sc_embeddings = jnp.zeros(
      (batch_size, canvas_length, embed_dim),
      dtype=jnp.bfloat16,
  )

  num_steps = (
      num_steps_override
      if num_steps_override is not None
      else max_denoising_steps
  )

  for step in range(num_steps):
    step_rng, sample_rng = jax.random.split(step_rng)
    current_noise_proportion = jnp.full(
        (batch_size,), 1.0 - step / max_denoising_steps
    )
    target_noise_proportion = jnp.full(
        (batch_size,), 1.0 - (step + 1) / max_denoising_steps
    )
    step_output = sampler.sample_step(
        canvas=canvas,
        sc_embeddings=sc_embeddings,
        cache=cache,
        positions=positions,
        attention_mask=attention_mask,
        current_noise_proportion=current_noise_proportion,
        target_noise_proportion=target_noise_proportion,
        params=params,
        rng=sample_rng,
    )
    canvas = step_output.sampled_tokens
    sc_embeddings = step_output.sc_embeddings

  return canvas


class SamplerTest(parameterized.TestCase):

  def test_sampler_step_runs(self):
    """Tests that DiffusionSampler.sample_step runs without errors.

    This is a basic smoke test for a single denoising step in isolation.
    Verifies that sample_step can be initialized and applied with minimal
    inputs (no cache, no positional encodings), and that the output shape
    matches [batch_size, canvas_length].
    """

    batch_size = 1
    canvas_length = 4
    vocab_size = _SMALL_CONFIG.num_embed
    embed_dim = _SMALL_CONFIG.embed_dim

    model = _models.DiffusionGemma_26B_A4B(
        config=_SMALL_CONFIG,
        self_conditioning_config=_SMALL_SC_CONFIG,
    )

    sampler = _sampler.DiffusionSampler(
        model=model,
        end_tokens=(99,),
        forbidden_tokens=None,
        sampling=_sampling.Greedy(),
        cache_length=16,
        special_tokens=None,
        diffusion_process=_sampler.DiffusionProcess(),
        logit_shaper=_sampler.AnnealingTemperatureShaperConfig().make(),
        sample_from_predictions=_sampler.SampleFromPredictions(
            text_vocab_size=vocab_size,
        ),
        canvas_length=canvas_length,
        max_denoising_steps=2,
        text_vocab_size=vocab_size,
    )

    rng = jax.random.PRNGKey(0)
    params = model.init(
        rngs=rng,
        tokens=jnp.ones((batch_size, canvas_length), dtype=jnp.int32),
        sc_embeddings=jnp.ones(
            (batch_size, canvas_length, embed_dim), dtype=jnp.bfloat16
        ),
        attention_mask=jnp.ones(
            (batch_size, canvas_length, canvas_length), dtype=jnp.bool_
        ),
        method=model.call_with_self_conditioning,
    )['params']

    output = sampler.sample_step(
        canvas=jnp.ones((batch_size, canvas_length), dtype=jnp.int32),
        sc_embeddings=jnp.ones(
            (batch_size, canvas_length, embed_dim), dtype=jnp.bfloat16
        ),
        cache=None,
        positions=None,
        attention_mask=jnp.ones(
            (batch_size, canvas_length, canvas_length), dtype=jnp.bool_
        ),
        current_noise_proportion=jnp.full(
            (batch_size,), 0.5, dtype=jnp.float32
        ),
        target_noise_proportion=jnp.full((batch_size,), 0.4, dtype=jnp.float32),
        params=params,
        rng=rng,
    )
    self.assertEqual(output.sampled_tokens.shape, (batch_size, canvas_length))

  @parameterized.named_parameters(
      dict(
          testcase_name='no_cache',
          use_cache=False,
      ),
      dict(
          testcase_name='with_cache',
          use_cache=True,
      ),
  )
  def test_sample_next_canvas(self, use_cache):
    """Tests that DiffusionSampler.sample_next_canvas runs end-to-end.

    Runs the full multi-step denoising loop that produces a complete canvas.
    Parameterized over two modes:
    - no_cache: The canvas is generated without a KV cache (standalone
      denoising, no prior context).
    - with_cache: The canvas is generated with a pre-allocated KV cache,
      simulating a conversational setting where prior tokens are cached.

    Args:
      use_cache: Whether to use a KV cache.
    """

    batch_size = 1
    canvas_length = 4
    cache_length = 10
    vocab_size = _SMALL_CONFIG.num_embed
    embed_dim = _SMALL_CONFIG.embed_dim

    model = _models.DiffusionGemma_26B_A4B(
        config=_SMALL_CONFIG,
        self_conditioning_config=_SMALL_SC_CONFIG,
    )

    if use_cache:
      cache = _SMALL_CONFIG.init_cache(
          batch_size=batch_size,
          dtype=jnp.bfloat16,
          cache_length=cache_length,
      )
    else:
      cache = None

    sampler = _sampler.DiffusionSampler(
        model=model,
        end_tokens=(99,),
        forbidden_tokens=None,
        sampling=_sampling.Greedy(),
        cache_length=cache_length,
        special_tokens=None,
        diffusion_process=_sampler.DiffusionProcess(),
        logit_shaper=_sampler.AnnealingTemperatureShaperConfig().make(),
        sample_from_predictions=_sampler.SampleFromPredictions(
            text_vocab_size=vocab_size,
        ),
        canvas_length=canvas_length,
        max_denoising_steps=2,
        text_vocab_size=vocab_size,
    )

    rng = jax.random.PRNGKey(0)
    params = model.init(
        rngs=rng,
        tokens=jnp.ones((batch_size, canvas_length), dtype=jnp.int32),
        sc_embeddings=jnp.ones(
            (batch_size, canvas_length, embed_dim), dtype=jnp.bfloat16
        ),
        attention_mask=jnp.ones(
            (batch_size, canvas_length, canvas_length), dtype=jnp.bool_
        ),
        method=model.call_with_self_conditioning,
    )['params']

    output = sampler.sample_next_canvas(
        canvas_length=canvas_length,
        max_denoising_steps=2,
        batch_size=batch_size,
        cache=cache,
        params=params,
        rng=rng,
    )
    self.assertEqual(output.shape, (batch_size, canvas_length))

  def test_append_tokens_to_cache(self):
    """Tests that append_tokens_to_cache correctly advances the cache index.

    Verifies two sequential appends: first inserting `canvas_length` tokens
    into an empty cache and checking the end_index advances to
    `canvas_length`, then inserting another batch and confirming it advances
    to `2 * canvas_length`. This ensures the cache write pointer accumulates
    correctly across multiple calls.
    """
    batch_size = 1
    canvas_length = 4
    cache_length = 16
    vocab_size = _SMALL_CONFIG.num_embed

    model = _models.DiffusionGemma_26B_A4B(
        config=_SMALL_CONFIG,
        self_conditioning_config=_SMALL_SC_CONFIG,
    )

    sampler = _sampler.DiffusionSampler(
        model=model,
        end_tokens=(99,),
        forbidden_tokens=None,
        sampling=_sampling.Greedy(),
        cache_length=cache_length,
        special_tokens=None,
        diffusion_process=_sampler.DiffusionProcess(),
        logit_shaper=_sampler.AnnealingTemperatureShaperConfig().make(),
        sample_from_predictions=_sampler.SampleFromPredictions(
            text_vocab_size=vocab_size,
        ),
        canvas_length=canvas_length,
        max_denoising_steps=2,
        text_vocab_size=vocab_size,
    )

    cache = _SMALL_CONFIG.init_cache(
        batch_size=batch_size,
        dtype=jnp.bfloat16,
        cache_length=cache_length,
    )

    tokens = jnp.ones((batch_size, canvas_length), dtype=jnp.int32)

    rng = jax.random.PRNGKey(0)
    params = model.init(
        rngs=rng,
        tokens=tokens,
        cache=cache,
        positions=jnp.arange(canvas_length)[None, :],
        attention_mask=jnp.ones(
            (batch_size, canvas_length, cache_length), dtype=jnp.bool_
        ),
    )['params']

    updated_cache = sampler.append_tokens_to_cache(
        tokens=tokens,
        cache=cache,
        params=params,
    )

    first_layer = list(updated_cache.values())[0]
    expected_end_index = jnp.full((batch_size,), canvas_length, dtype=jnp.int32)
    np.testing.assert_array_equal(first_layer['end_index'], expected_end_index)

    updated_cache_2 = sampler.append_tokens_to_cache(
        tokens=tokens,
        cache=updated_cache,
        params=params,
    )
    first_layer_2 = list(updated_cache_2.values())[0]
    expected_end_index_2 = jnp.full(
        (batch_size,), 2 * canvas_length, dtype=jnp.int32
    )
    np.testing.assert_array_equal(
        first_layer_2['end_index'], expected_end_index_2
    )

  def test_make_global_attention_no_cache(self):
    """Tests that the global attention mask is all-ones when no cache is used.

    Without a cache, the mask should be a simple [batch, canvas, canvas]
    tensor of ones, allowing full bidirectional self-attention.
    """
    mask = _sampler._make_global_attention_mask(
        batch_size=2,
        canvas_length=4,
        cache_length=None,
        num_valid_tokens=None,
    )
    np.testing.assert_array_equal(mask, jnp.ones((2, 4, 4), dtype=jnp.bool_))

  def test_make_global_attention_mask_batched_edge_cases(self):
    """Tests the global attention mask with a KV cache across edge cases.

    The mask shape is [batch, canvas_length, cache_length] with 1s for
    valid positions and 0s for padding. All canvas tokens attend to the
    same set of valid positions (no causal constraint). Edge cases tested
    as a single batched call:
    - no_wrap: Partially filled cache (3 of 10 slots); total valid
      positions = 3 + 4 = 7, so positions 7-9 are masked out.
    - with_wrap: Nearly full cache (8 valid); total valid = min(12, 10)
      = 10, so all positions are visible.
    - empty_cache: Zero valid cache tokens; only the 4 canvas tokens are
      visible, positions 4-9 are masked out.
    - overfull_cache: More tokens written than cache_length (12 > 10);
      all positions should be visible since the cache is saturated.
    """
    # cache_length=10, canvas_length=4
    # Test multiple global mask edge cases in a single batched call
    edge_cases = [
        (
            'no_wrap',
            3,
            jnp.array([
                # total_valid = min(3+4, 10) = 7
                # mask should be True for 0..6, False for 7..9
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            ]).reshape(4, 10),
        ),
        (
            'with_wrap',
            8,
            jnp.array([
                # total_valid = min(8+4, 10) = 10
                # mask should be all True
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]).reshape(4, 10),
        ),
        (
            'empty_cache',
            0,
            jnp.array([
                # total_valid = min(0+4, 10) = 4
                # mask should be True for 0..3, False for 4..9
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            ]).reshape(4, 10),
        ),
        (
            'overfull_cache',
            12,
            jnp.array([
                # total_valid = min(12+4, 10) = 10
                # mask should be all True
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]).reshape(4, 10),
        ),
    ]

    num_valid_cache_tokens = jnp.array([case[1] for case in edge_cases])

    mask = _sampler._make_global_attention_mask(
        batch_size=len(edge_cases),
        canvas_length=4,
        cache_length=10,
        num_valid_tokens=num_valid_cache_tokens,
    )

    for i, (name, _, expected) in enumerate(edge_cases):
      with self.subTest(name=name):
        np.testing.assert_array_equal(mask[i], expected)

  def test_make_causal_attention_mask_no_cache(self):
    """Tests that the causal mask is lower-triangular when no cache is used.

    Without a cache, the mask should be a standard lower-triangular matrix
    broadcast across the batch dimension, enforcing standard autoregressive
    left-to-right attention.
    """
    mask = _sampler._make_causal_attention_mask(
        batch_size=2,
        canvas_length=7,
        cache_length=None,
        num_valid_cache_tokens=None,
    )
    expected = jnp.tril(jnp.ones((7, 7), dtype=jnp.bool_))
    expected = jnp.broadcast_to(expected[jnp.newaxis, :, :], (2, 7, 7))
    np.testing.assert_array_equal(mask, expected)

  def test_make_causal_attention_mask_batched_edge_cases(self):
    """Tests the causal attention mask for cache insertion across edge cases.

    When appending canvas tokens to a ring-buffer KV cache, the mask must
    handle wrap-around correctly. Each canvas token should attend to all
    previously cached tokens plus all preceding tokens in the current
    canvas, but NOT future canvas tokens. Edge cases tested:
    - no_wrap: 2 valid cache tokens; new tokens write to indices 2-5,
      no wrap-around occurs.
    - exactly_full: 6 valid tokens; the last canvas token writes to
      index 9, exactly filling the cache with no wrap.
    - with_wrap: 8 valid tokens; writes start at index 8 and wrap to
      indices 0-1, testing ring-buffer wrap-around.
    - full_cache: 10 valid tokens (cache already full); writes wrap
      starting at index 0, overwriting the oldest entries.
    - overfull_cache: 12 valid tokens (more than cache capacity); writes
      start at index 2 (12 % 10), also wrapping, verifying the
      modular arithmetic is correct when the counter exceeds capacity.
    """
    # cache_length=10, seq_len=4
    # Test multiple edge cases in a single batched call
    edge_cases = [
        (
            'no_wrap',
            2,
            jnp.array([
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            ]),
        ),
        (
            'exactly_full',
            6,
            jnp.array([
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
        ),
        (
            'with_wrap',
            8,
            jnp.array([
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
        ),
        (
            'full_cache',
            10,
            jnp.array([
                [1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
        ),
        (
            'overfull_cache',
            12,
            jnp.array([
                [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
        ),
    ]

    num_valid_cache_tokens = jnp.array([case[1] for case in edge_cases])

    mask = _sampler._make_causal_attention_mask(
        batch_size=len(edge_cases),
        canvas_length=4,
        cache_length=10,
        num_valid_cache_tokens=num_valid_cache_tokens,
    )

    for i, (name, _, expected) in enumerate(edge_cases):
      with self.subTest(name=name):
        np.testing.assert_array_equal(mask[i], expected)

  def test_diffusion_sample_step_runs(self):
    """Tests the DiffusionSamplerLoop._sample_step integration.

    Unlike test_sampler_step_runs which tests the raw DiffusionSampler,
    this tests the full sampling pipeline through DiffusionSamplerLoop,
    which orchestrates a canvas generation, stop-token truncation, and
    cache update in a single step. Verifies that:
    - predicted_tokens has the correct shape.
    - The step counter advances by canvas_length.
    - The batch is not prematurely marked as done.
    - The cache end_index advances by canvas_length.
    """
    batch_size = 1
    canvas_length = 4
    cache_length = 16
    max_out_length = 16
    max_denoising_steps = 2
    vocab_size = _SMALL_CONFIG.num_embed
    embed_dim = _SMALL_CONFIG.embed_dim

    model = _models.DiffusionGemma_26B_A4B(
        config=_SMALL_CONFIG,
        self_conditioning_config=_SMALL_SC_CONFIG,
    )

    sampler_loop = _sampler.DiffusionSampler(
        model=model,
        end_tokens=(99,),
        forbidden_tokens=None,
        sampling=_sampling.Greedy(),
        cache_length=cache_length,
        special_tokens=None,
        diffusion_process=_sampler.DiffusionProcess(),
        logit_shaper=_sampler.AnnealingTemperatureShaperConfig().make(),
        sample_from_predictions=_sampler.SampleFromPredictions(
            text_vocab_size=vocab_size,
        ),
        canvas_length=canvas_length,
        max_denoising_steps=max_denoising_steps,
        text_vocab_size=vocab_size,
    )

    cache = _SMALL_CONFIG.init_cache(
        batch_size=batch_size,
        dtype=jnp.bfloat16,
        cache_length=cache_length,
    )

    rng = jax.random.PRNGKey(0)
    init_rng, step_rng = jax.random.split(rng)

    params = model.init(
        rngs=init_rng,
        tokens=jnp.ones((batch_size, canvas_length), dtype=jnp.int32),
        sc_embeddings=jnp.ones(
            (batch_size, canvas_length, embed_dim), dtype=jnp.bfloat16
        ),
        attention_mask=jnp.ones(
            (batch_size, canvas_length, canvas_length), dtype=jnp.bool_
        ),
        method=model.call_with_self_conditioning,
    )['params']

    init_state = _sampler_loop.SamplingState(
        step=jnp.int32(0),
        done=jnp.zeros((batch_size,), dtype=jnp.bool_),
        last_token=jnp.zeros((batch_size,), dtype=jnp.int32),
        last_token_pos=jnp.zeros((batch_size,), dtype=jnp.int32),
        predicted_tokens=jnp.zeros(
            (batch_size, max_out_length), dtype=jnp.int32
        ),
        cache=cache,
        rng=step_rng,
        init_cache_length=jnp.int32(0),
        full_attention_mask=jnp.ones(
            (batch_size, cache_length), dtype=jnp.bool_
        ),
    )

    with jax.disable_jit():
      new_state = sampler_loop._sample_step(
          init_state,
          params=params,
      )

    self.assertEqual(
        new_state.predicted_tokens.shape, (batch_size, max_out_length)
    )
    self.assertEqual(new_state.step, canvas_length)
    self.assertFalse(jnp.all(new_state.done))

    cache_layer = list(new_state.cache.values())[0]
    expected_end_index = jnp.full((batch_size,), canvas_length, dtype=jnp.int32)
    np.testing.assert_array_equal(cache_layer['end_index'], expected_end_index)

  def test_chat_sampler_runs(self):
    """Tests the full ChatSampler end-to-end with a mock tokenizer.

    Exercises the complete inference stack: ChatSampler -> Sampler ->
    DiffusionSamplerLoop -> DiffusionSampler, including tokenization,
    prompt encoding, multi-step diffusion sampling, and decoding back to
    a string. Uses a mock tokenizer that maps characters to token IDs
    via their ordinal values (mod vocab_size). Verifies that the output
    is a valid string.
    """
    canvas_length = 4
    cache_length = 512
    max_out_length = 16
    max_denoising_steps = 2
    vocab_size = _SMALL_CONFIG.num_embed
    embed_dim = _SMALL_CONFIG.embed_dim

    model = _models.DiffusionGemma_26B_A4B(
        config=_SMALL_CONFIG,
        self_conditioning_config=_SMALL_SC_CONFIG,
    )

    rng = jax.random.PRNGKey(0)

    params = model.init(
        rngs=rng,
        tokens=jnp.ones((1, canvas_length), dtype=jnp.int32),
        sc_embeddings=jnp.ones(
            (1, canvas_length, embed_dim), dtype=jnp.bfloat16
        ),
        attention_mask=jnp.ones(
            (1, canvas_length, canvas_length), dtype=jnp.bool_
        ),
        method=model.call_with_self_conditioning,
    )['params']

    tokenizer = _MockTokenizer(vocab_size=vocab_size)

    chat_sampler = _chat_sampler.ChatSampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
        diffusion_process=_sampler.DiffusionProcess(),
        logit_shaper=_sampler.AnnealingTemperatureShaperConfig().make(),
        sample_from_predictions=_sampler.SampleFromPredictions(
            text_vocab_size=vocab_size,
        ),
        canvas_length=canvas_length,
        max_denoising_steps=max_denoising_steps,
        cache_length=cache_length,
        max_out_length=max_out_length,
    )

    with jax.disable_jit():
      output = chat_sampler.chat('hello')

    self.assertIsInstance(output, str)

  def test_chat_sampler_gemma4_dispatch(self):
    """Tests that diffusion ChatSampler works when _is_gemma4 is True."""
    canvas_length = 4
    cache_length = 512
    max_out_length = 16
    max_denoising_steps = 2
    vocab_size = _SMALL_CONFIG.num_embed
    embed_dim = _SMALL_CONFIG.embed_dim

    class _DiffusionGemmaTransformer(_models.DiffusionGemma_26B_A4B):
      INFO = _gemma4_transformer.ModelInfo(tokenizer_version=4)

    model = _DiffusionGemmaTransformer(
        config=_SMALL_CONFIG,
        self_conditioning_config=_SMALL_SC_CONFIG,
    )

    rng = jax.random.PRNGKey(0)

    params = model.init(
        rngs=rng,
        tokens=jnp.ones((1, canvas_length), dtype=jnp.int32),
        sc_embeddings=jnp.ones(
            (1, canvas_length, embed_dim), dtype=jnp.float32
        ),
        attention_mask=jnp.ones(
            (1, canvas_length, canvas_length), dtype=jnp.bool_
        ),
        method=model.call_with_self_conditioning,
    )['params']

    tokenizer = _MockTokenizer(vocab_size=vocab_size)

    chat_sampler = _chat_sampler.ChatSampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
        diffusion_process=_sampler.DiffusionProcess(),
        logit_shaper=_sampler.AnnealingTemperatureShaperConfig().make(),
        sample_from_predictions=_sampler.SampleFromPredictions(
            text_vocab_size=vocab_size,
        ),
        canvas_length=canvas_length,
        max_denoising_steps=max_denoising_steps,
        cache_length=cache_length,
        max_out_length=max_out_length,
    )

    # Verify _is_gemma4 is True — this is the condition that triggers the bug.
    self.assertTrue(chat_sampler._is_gemma4)  # pylint: disable=protected-access

    # Verify the diffusion sampler is used (not Gemma4Sampler) by tracking
    # calls to sample_next_canvas, which only DiffusionSampler invokes.
    original = _sampler.DiffusionSampler.sample_next_canvas
    sample_next_canvas_called = False

    def tracking_sample_next_canvas(*args, **kwargs):
      nonlocal sample_next_canvas_called
      sample_next_canvas_called = True
      return original(*args, **kwargs)

    with mock.patch.object(
        _sampler.DiffusionSampler,
        'sample_next_canvas',
        new=tracking_sample_next_canvas,
    ):
      with jax.disable_jit():
        output = chat_sampler.chat('hello')

    self.assertIsInstance(output, str)
    self.assertTrue(sample_next_canvas_called)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_cache',
          use_cache=False,
      ),
      dict(
          testcase_name='with_cache',
          use_cache=True,
      ),
  )
  def test_sample_next_canvas_while_loop_matches_for_loop(self, use_cache):
    """Verifies while_loop-based denoising is numerically identical to a for loop.

    Runs sample_next_canvas (which uses jax.lax.while_loop internally) and an
    explicit for-loop reference that calls sample_step sequentially. Asserts
    the final token outputs are identical, catching any silent numerical
    divergence introduced by the while_loop refactor.

    Args:
      use_cache: Whether to use a KV cache.
    """
    batch_size = 2
    canvas_length = 4
    cache_length = 16
    max_denoising_steps = 8
    vocab_size = _SMALL_CONFIG.num_embed
    embed_dim = _SMALL_CONFIG.embed_dim

    model = _models.DiffusionGemma_26B_A4B(
        config=_SMALL_CONFIG,
        self_conditioning_config=_SMALL_SC_CONFIG,
    )

    sampler = _sampler.DiffusionSampler(
        model=model,
        end_tokens=(99,),
        forbidden_tokens=None,
        sampling=_sampling.Greedy(),
        cache_length=cache_length,
        special_tokens=None,
        diffusion_process=_sampler.DiffusionProcess(),
        logit_shaper=_sampler.AnnealingTemperatureShaperConfig().make(),
        sample_from_predictions=_sampler.SampleFromPredictions(
            text_vocab_size=vocab_size,
        ),
        canvas_length=canvas_length,
        max_denoising_steps=max_denoising_steps,
        text_vocab_size=vocab_size,
    )

    if use_cache:
      cache = _SMALL_CONFIG.init_cache(
          batch_size=batch_size,
          dtype=jnp.bfloat16,
          cache_length=cache_length,
      )
    else:
      cache = None

    rng = jax.random.PRNGKey(42)
    params = model.init(
        rngs=rng,
        tokens=jnp.ones((batch_size, canvas_length), dtype=jnp.int32),
        sc_embeddings=jnp.ones(
            (batch_size, canvas_length, embed_dim), dtype=jnp.bfloat16
        ),
        attention_mask=jnp.ones(
            (batch_size, canvas_length, canvas_length), dtype=jnp.bool_
        ),
        method=model.call_with_self_conditioning,
    )['params']

    while_loop_output = _sample_with_while_loop(
        sampler,
        params,
        cache,
        batch_size,
        canvas_length,
        max_denoising_steps,
        rng,
    )
    for_loop_output = _sample_with_for_loop(
        sampler,
        params,
        cache,
        batch_size,
        canvas_length,
        embed_dim,
        vocab_size,
        max_denoising_steps,
        rng,
    )

    np.testing.assert_array_equal(while_loop_output, for_loop_output)

  def test_early_stopping_terminates_after_one_step(self):
    """Tests that an EarlyStopFn that always returns True stops after step 0.

    Creates a sampler with an early_stop_fn that returns True unconditionally.
    Compares the output against a manual 1-iteration for-loop reference that
    uses the same noise schedule (max_denoising_steps=8), so the noise
    proportions at step 0 are identical.
    """

    class AlwaysStop:

      def should_stop(self, *, step, canvas, previous_canvas, logits):
        del step, canvas, previous_canvas, logits
        return jnp.bool_(True)

    batch_size = 2
    canvas_length = 4
    cache_length = 16
    max_denoising_steps = 8
    vocab_size = _SMALL_CONFIG.num_embed
    embed_dim = _SMALL_CONFIG.embed_dim

    model = _models.DiffusionGemma_26B_A4B(
        config=_SMALL_CONFIG,
        self_conditioning_config=_SMALL_SC_CONFIG,
    )

    rng = jax.random.PRNGKey(42)
    params = model.init(
        rngs=rng,
        tokens=jnp.ones((batch_size, canvas_length), dtype=jnp.int32),
        sc_embeddings=jnp.ones(
            (batch_size, canvas_length, embed_dim), dtype=jnp.float32
        ),
        attention_mask=jnp.ones(
            (batch_size, canvas_length, canvas_length), dtype=jnp.bool_
        ),
        method=model.call_with_self_conditioning,
    )['params']

    cache = _SMALL_CONFIG.init_cache(
        batch_size=batch_size,
        dtype=jnp.float32,
        cache_length=cache_length,
    )

    sampler_early = _sampler.DiffusionSampler(
        model=model,
        end_tokens=(99,),
        forbidden_tokens=None,
        sampling=_sampling.Greedy(),
        cache_length=cache_length,
        special_tokens=None,
        diffusion_process=_sampler.DiffusionProcess(),
        logit_shaper=_sampler.AnnealingTemperatureShaperConfig().make(),
        sample_from_predictions=_sampler.SampleFromPredictions(
            text_vocab_size=vocab_size,
        ),
        canvas_length=canvas_length,
        max_denoising_steps=max_denoising_steps,
        early_stop_fn=AlwaysStop(),
        text_vocab_size=vocab_size,
    )

    early_output = sampler_early.sample_next_canvas(
        canvas_length=canvas_length,
        max_denoising_steps=max_denoising_steps,
        batch_size=batch_size,
        cache=cache,
        params=params,
        rng=rng,
    )

    # Reference: run 1 iteration of the for-loop with the SAME noise schedule
    # (max_denoising_steps=8, but only 1 step executed).
    ref_output = _sample_with_for_loop(
        sampler_early,
        params,
        cache,
        batch_size,
        canvas_length,
        embed_dim,
        vocab_size,
        max_denoising_steps=max_denoising_steps,
        rng=rng,
        num_steps_override=1,
    )

    np.testing.assert_array_equal(early_output, ref_output)

  def test_token_stability_early_stop(self):
    """Tests that TokenStabilityEarlyStop runs end-to-end and produces valid output.

    Uses the built-in TokenStabilityEarlyStop to verify it integrates correctly
    with the while_loop. Also checks determinism: two calls with the same
    RNG produce identical results.
    """
    batch_size = 1
    canvas_length = 4
    cache_length = 16
    max_denoising_steps = 10
    vocab_size = _SMALL_CONFIG.num_embed
    embed_dim = _SMALL_CONFIG.embed_dim

    model = _models.DiffusionGemma_26B_A4B(
        config=_SMALL_CONFIG,
        self_conditioning_config=_SMALL_SC_CONFIG,
    )

    rng = jax.random.PRNGKey(123)
    params = model.init(
        rngs=rng,
        tokens=jnp.ones((batch_size, canvas_length), dtype=jnp.int32),
        sc_embeddings=jnp.ones(
            (batch_size, canvas_length, embed_dim), dtype=jnp.float32
        ),
        attention_mask=jnp.ones(
            (batch_size, canvas_length, canvas_length), dtype=jnp.bool_
        ),
        method=model.call_with_self_conditioning,
    )['params']

    cache = _SMALL_CONFIG.init_cache(
        batch_size=batch_size,
        dtype=jnp.float32,
        cache_length=cache_length,
    )

    sampler_stability = _sampler.DiffusionSampler(
        model=model,
        end_tokens=(99,),
        forbidden_tokens=None,
        sampling=_sampling.Greedy(),
        cache_length=cache_length,
        special_tokens=None,
        diffusion_process=_sampler.DiffusionProcess(),
        logit_shaper=_sampler.AnnealingTemperatureShaperConfig().make(),
        sample_from_predictions=_sampler.SampleFromPredictions(
            text_vocab_size=vocab_size,
        ),
        canvas_length=canvas_length,
        max_denoising_steps=max_denoising_steps,
        early_stop_fn=_early_stopping.TokenStabilityEarlyStop(),
        text_vocab_size=vocab_size,
    )

    output1 = sampler_stability.sample_next_canvas(
        canvas_length=canvas_length,
        max_denoising_steps=max_denoising_steps,
        batch_size=batch_size,
        cache=cache,
        params=params,
        rng=rng,
    )
    output2 = sampler_stability.sample_next_canvas(
        canvas_length=canvas_length,
        max_denoising_steps=max_denoising_steps,
        batch_size=batch_size,
        cache=cache,
        params=params,
        rng=rng,
    )

    self.assertEqual(output1.shape, (batch_size, canvas_length))
    np.testing.assert_array_equal(output1, output2)


if __name__ == '__main__':
  absltest.main()
