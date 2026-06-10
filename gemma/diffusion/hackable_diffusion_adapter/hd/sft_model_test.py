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

"""Unit tests for SFT encoder/decoder masking behaviour.

Tests verify:
1. Encoder and denoiser logits are invariant to the number of PAD tokens.
2. Encoder logits at position i are invariant to prompt tokens at j > i
(causal).
3. Denoiser logits change when the prompt changes.
4. Encoder and decoder losses produce non-zero gradients (including through
LoRA).
"""

from absl.testing import absltest

import flax.linen as nn
from gemma.diffusion import _models
from gemma.diffusion import _transformer as diffusion_transformer
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network
from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model
from gemma.gm.nn.gemma4 import _config
from gemma.gm.nn.gemma4 import _modules
from hackable_diffusion.lib.corruption import discrete
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.training import time_sampling
import jax
import jax.numpy as jnp
import numpy as np
import optax



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


def _make_gemma_network():
  """Create a WrappedDiffusionGemmaNetwork with a tiny Gemma model."""
  small_config = _make_small_config()
  gemma_model = _models.DiffusionGemma_A26B_A4B(
      config=small_config,
      self_conditioning_config=diffusion_transformer.SelfConditioningConfig(
          features=small_config.embed_dim,
          hidden_dim=small_config.hidden_dim,
      ),
  )
  return hd_gemma_network.WrappedDiffusionGemmaNetwork(
      gemma_model=gemma_model,
  )


def _sft_encode(
    gemma_network,
    *,
    prompt,
    x0_tokens,
    canvas_mask,
    selected_canvas_idx,
    prompt_len,
    total_canvas_len,
    canvas_size,
    pad_token=0,
):
  """Prefills the Gemma KV cache using the SFT encoder sequence."""
  del total_canvas_len  # Unused; accepted for API consistency.
  x0_tokens = x0_tokens[..., 0] if x0_tokens.ndim == 3 else x0_tokens
  full_seq = jnp.concatenate([prompt, x0_tokens], axis=1)
  prompt_mask = prompt != pad_token
  full_seq_mask = jnp.concatenate([prompt_mask, canvas_mask], axis=1)

  kv_cache, encoder_logits, positions, _ = (
      hd_gemma_network.prefill_kv_cache_with_encoder(
          tokens=full_seq,
          input_mask=full_seq_mask,
          init_cache_fn=gemma_network.init_cache,
          encoder_fn=gemma_network.encoder_call,
      )
  )

  end_index = prompt_len + selected_canvas_idx * canvas_size
  kv_cache = mask_helpers.set_cache_end_index(kv_cache, end_index)

  return encoder_logits, kv_cache, positions, prompt_mask


def _init_and_run(
    rng,
    prompt,
    x0,
    canvas_mask,
    xt,
    prompt_len,
    canvas_size=4,
    selected_canvas_idx=None,
    variables=None,
):
  """Initialize (if needed) and run sft_encode + sft_decode.

  Args:
    rng: PRNG key.
    prompt: Prompt tokens [B, PromptLen].
    x0: Clean canvas tokens [B, TotalCanvasLen].
    canvas_mask: Valid-canvas mask [B, TotalCanvasLen].
    xt: Noised canvas tokens [B, TotalCanvasLen, 1].
    prompt_len: Fixed prompt length.
    canvas_size: Tokens per canvas.
    selected_canvas_idx: Per-example selected canvas index [B]. Defaults to
      zeros (canvas 0) if not provided.
    variables: If provided, reuse these params; otherwise init from scratch.

  Returns:
    (output_dict, variables) where output_dict has keys:
      'encoder_logits', 'logits', 'kv_cache'.
  """
  gemma_network = _make_gemma_network()
  total_canvas_len = x0.shape[1]
  batch_size = prompt.shape[0]
  time = jnp.ones((batch_size, total_canvas_len, 1), dtype=jnp.float32)
  if selected_canvas_idx is None:
    selected_canvas_idx = jnp.zeros((batch_size,), dtype=jnp.int32)

  # We need to init+run through a thin wrapper so Flax can bind the submodule.
  class _TestWrapper(nn.Module):
    gemma_network: hd_gemma_network.WrappedDiffusionGemmaNetwork

    @nn.compact
    def __call__(
        self,
        prompt,
        x0,
        canvas_mask,
        selected_canvas_idx,
        xt,
        time,
        prompt_len,
        total_canvas_len,
        canvas_size,
    ):
      encoder_logits, kv_cache, positions, prompt_mask = _sft_encode(
          gemma_network=self.gemma_network,
          prompt=prompt,
          x0_tokens=x0,
          canvas_mask=canvas_mask,
          selected_canvas_idx=selected_canvas_idx,
          prompt_len=prompt_len,
          total_canvas_len=total_canvas_len,
          canvas_size=canvas_size,
      )
      denoiser_output = sft_model.sft_decode(
          gemma_network=self.gemma_network,
          xt=xt,
          time=time,
          kv_cache=kv_cache,
          positions=positions,
          prompt_mask=prompt_mask,
          canvas_mask=canvas_mask,
          selected_canvas_idx=selected_canvas_idx,
          prompt_len=prompt_len,
          total_canvas_len=total_canvas_len,
          canvas_size=canvas_size,
          is_training=False,
      )
      return {
          'encoder_logits': encoder_logits,
          'logits': denoiser_output['logits'],
          'kv_cache': kv_cache,
      }

  wrapper = _TestWrapper(gemma_network=gemma_network)

  if variables is None:
    variables = wrapper.init(
        rng,
        prompt=prompt,
        x0=x0,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        xt=xt,
        time=time,
        prompt_len=prompt_len,
        total_canvas_len=total_canvas_len,
        canvas_size=canvas_size,
    )

  output = wrapper.apply(
      variables,
      prompt=prompt,
      x0=x0,
      canvas_mask=canvas_mask,
      selected_canvas_idx=selected_canvas_idx,
      xt=xt,
      time=time,
      prompt_len=prompt_len,
      total_canvas_len=total_canvas_len,
      canvas_size=canvas_size,
  )
  return output, variables


def _make_sft_diffusion(
    prompt_len,
    canvas_size,
    num_canvases=1,
    num_embed=32,
):
  """Create an SFTDiffusion module with a tiny Gemma model."""
  corruption_process = discrete.CategoricalProcess.uniform_process(
      num_categories=num_embed,
      schedule=schedules.RFSchedule(),
  )
  ts = time_sampling.UniformTimeSampler()
  gemma_network = _make_gemma_network()
  return sft_model.SFTDiffusion(
      gemma_network=gemma_network,
      corruption_process=corruption_process,
      time_sampler=ts,
      prompt_len=prompt_len,
      canvas_size=canvas_size,
      num_canvases=num_canvases,
      # Kontext keys are unused in direct __call__ tests.
      x0='unused',
      prompt='unused',
      canvas_id='unused',
      canvas_mask='unused',
      encoder_target='unused',
      encoder_target_mask='unused',
      stop_gradient_from_denoiser_to_encoder=False,
      self_cond_prob=0.0,
  )


def _init_and_run_sft_diffusion(
    rng,
    prompt,
    x0,
    canvas_mask,
    canvas_id,
    encoder_target,
    encoder_target_mask,
    prompt_len,
    canvas_size=4,
    num_canvases=1,
    variables=None,
):
  """Initialize (if needed) and run SFTDiffusion.__call__.

  Args:
    rng: PRNG key.
    prompt: Prompt tokens [B, PromptLen].
    x0: Canvas tokens [B, TotalCanvasLen, 1].
    canvas_mask: Valid-canvas mask [B, TotalCanvasLen].
    canvas_id: Canvas IDs [B, TotalCanvasLen].
    encoder_target: AR target for encoder [B, FullSeqLen].
    encoder_target_mask: Mask for encoder target [B, FullSeqLen].
    prompt_len: Fixed prompt length.
    canvas_size: Tokens per canvas.
    num_canvases: Number of canvases.
    variables: If provided, reuse these params; otherwise init from scratch.

  Returns:
    (output_dict, variables) where output_dict is the return value of
    SFTDiffusion.__call__.
  """
  model = _make_sft_diffusion(
      prompt_len=prompt_len,
      canvas_size=canvas_size,
      num_canvases=num_canvases,
  )

  rngs = {'params': rng, 'sampling': rng}

  if variables is None:
    variables = model.init(
        rngs,
        x0=x0,
        prompt=prompt,
        canvas_id=canvas_id,
        canvas_mask=canvas_mask,
        encoder_target=encoder_target,
        encoder_target_mask=encoder_target_mask,
        is_training=False,
    )

  output = model.apply(
      variables,
      x0=x0,
      prompt=prompt,
      canvas_id=canvas_id,
      canvas_mask=canvas_mask,
      encoder_target=encoder_target,
      encoder_target_mask=encoder_target_mask,
      is_training=False,
      rngs={'sampling': rng},
  )
  return output, variables


class SFTDiffusionCallTest(absltest.TestCase):
  """Tests for SFTDiffusion.__call__ (full forward pass).

  Mirrors the tests in SFTEncodeDecodeTest but exercises the full
  SFTDiffusion.__call__ code path, which includes time sampling, corruption,
  canvas selection, encoder prefill, self-conditioning, and decoder passes.
  """

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(42)

  def _make_inputs(self, prompt, canvas_size=4, num_canvases=1):
    """Build matching x0, canvas_mask, canvas_id, encoder_target/mask."""
    prompt_len = prompt.shape[1]
    total_canvas_len = canvas_size * num_canvases
    batch_size = prompt.shape[0]
    # Use simple sequential tokens for the canvas.
    x0_flat = jnp.tile(
        jnp.arange(7, 7 + canvas_size, dtype=jnp.int32),
        (batch_size, num_canvases),
    )  # [B, TotalCanvasLen]
    x0 = x0_flat[..., None]  # [B, TotalCanvasLen, 1]
    canvas_mask = jnp.ones((batch_size, total_canvas_len), dtype=jnp.bool_)
    canvas_id = jnp.repeat(
        jnp.arange(num_canvases, dtype=jnp.int32)[None, :],
        canvas_size,
        axis=1,
    )
    canvas_id = jnp.broadcast_to(canvas_id, (batch_size, total_canvas_len))
    # Encoder target: shifted full sequence.
    full_seq_len = prompt_len + total_canvas_len
    encoder_target = jnp.concatenate(
        [prompt[:, 1:], x0_flat, jnp.zeros((batch_size, 1), dtype=jnp.int32)],
        axis=1,
    )[:, :full_seq_len]
    encoder_target_mask = jnp.ones(
        (batch_size, full_seq_len), dtype=jnp.float32
    )
    return x0, canvas_mask, canvas_id, encoder_target, encoder_target_mask

  def test_padding_invariance(self):
    """Encoder logits at real prompt positions should be invariant to padding.

    We use two runs with different prompt_len to simulate:
      Input 1: [PROMPT(3 real), PAD(3), RESPONSE(4)]  (prompt_len=6)
      Input 2: [PROMPT(3 real), PAD(5), RESPONSE(4)]  (prompt_len=8)

    Encoder logits at the first 3 positions (real prompt) should match.
    Denoiser logits should match (same real content, same rng).
    """
    canvas_size = 4

    prompt_1 = jnp.array([[5, 10, 15, 0, 0, 0]], dtype=jnp.int32)
    prompt_2 = jnp.array([[5, 10, 15, 0, 0, 0, 0, 0]], dtype=jnp.int32)

    x0_1, cm_1, cid_1, et_1, etm_1 = self._make_inputs(
        prompt_1, canvas_size=canvas_size
    )
    x0_2, cm_2, cid_2, et_2, etm_2 = self._make_inputs(
        prompt_2, canvas_size=canvas_size
    )

    out_1, variables_1 = _init_and_run_sft_diffusion(
        self.rng,
        prompt_1,
        x0_1,
        cm_1,
        cid_1,
        et_1,
        etm_1,
        prompt_len=6,
        canvas_size=canvas_size,
    )
    out_2, variables_2 = _init_and_run_sft_diffusion(
        self.rng,
        prompt_2,
        x0_2,
        cm_2,
        cid_2,
        et_2,
        etm_2,
        prompt_len=8,
        canvas_size=canvas_size,
    )

    # Assert that variables are the same (same rng -> same init).
    jax.tree_util.tree_map(np.testing.assert_allclose, variables_1, variables_2)

    # Encoder logits at the first 3 positions (real prompt) should match.
    enc_logits_1 = out_1['encoder_logits'][:, :3, :]
    enc_logits_2 = out_2['encoder_logits'][:, :3, :]
    np.testing.assert_allclose(
        enc_logits_1,
        enc_logits_2,
        atol=1e-4,
        err_msg=(
            'SFTDiffusion: Encoder logits should be invariant to padding for'
            ' real tokens'
        ),
    )

    # Denoiser logits should be the same since real prompt and response
    # tokens are identical and rngs match.
    den_logits_1 = out_1['output']['logits']
    den_logits_2 = out_2['output']['logits']
    np.testing.assert_allclose(
        den_logits_1,
        den_logits_2,
        atol=1e-4,
        err_msg='SFTDiffusion: Denoiser logits should be invariant to padding',
    )

  def test_encoder_causality(self):
    """Encoder logits at position i should be invariant to tokens at j > i.

    Input 1: [tok1=5, tok2=10, tok3=15, tok4=20, PAD, PAD, RESPONSE...]
    Input 2: [tok1=5, tok2=10, tok3=25, tok4=30, PAD, PAD, RESPONSE...]

    Encoder logits at positions 0 and 1 should be the same.
    Encoder logits at positions 2+ should differ.
    """
    canvas_size = 4
    prompt_len = 6

    prompt_1 = jnp.array([[5, 10, 15, 20, 0, 0]], dtype=jnp.int32)
    prompt_2 = jnp.array([[5, 10, 25, 30, 0, 0]], dtype=jnp.int32)

    x0_1, cm_1, cid_1, et_1, etm_1 = self._make_inputs(
        prompt_1, canvas_size=canvas_size
    )
    x0_2, cm_2, cid_2, et_2, etm_2 = self._make_inputs(
        prompt_2, canvas_size=canvas_size
    )

    out_1, variables = _init_and_run_sft_diffusion(
        self.rng,
        prompt_1,
        x0_1,
        cm_1,
        cid_1,
        et_1,
        etm_1,
        prompt_len=prompt_len,
        canvas_size=canvas_size,
    )
    out_2, _ = _init_and_run_sft_diffusion(
        self.rng,
        prompt_2,
        x0_2,
        cm_2,
        cid_2,
        et_2,
        etm_2,
        prompt_len=prompt_len,
        canvas_size=canvas_size,
        variables=variables,
    )

    enc_1 = out_1['encoder_logits']
    enc_2 = out_2['encoder_logits']

    # Positions 0 and 1 see the same tokens (5, 10) so logits match.
    np.testing.assert_allclose(
        enc_1[:, :2, :],
        enc_2[:, :2, :],
        atol=1e-4,
        err_msg=(
            'SFTDiffusion: Encoder logits at early positions should be'
            ' invariant to later tokens'
        ),
    )

    # Position 2 sees different tokens, so logits should differ.
    diff_at_2 = jnp.abs(enc_1[:, 2, :] - enc_2[:, 2, :]).max()
    self.assertGreater(
        float(diff_at_2),
        1e-3,
        'SFTDiffusion: Encoder logits at position 2 should differ when token'
        ' 2 changes',
    )

  def test_denoiser_sensitivity_to_prompt(self):
    """Denoiser logits should change when the prompt changes.

    Same response tokens but different prompt -> denoiser logits must differ.
    """
    canvas_size = 4
    prompt_len = 6

    prompt_1 = jnp.array([[5, 10, 15, 20, 0, 0]], dtype=jnp.int32)
    prompt_2 = jnp.array([[5, 10, 25, 30, 0, 0]], dtype=jnp.int32)

    x0_1, cm_1, cid_1, et_1, etm_1 = self._make_inputs(
        prompt_1, canvas_size=canvas_size
    )
    x0_2, cm_2, cid_2, et_2, etm_2 = self._make_inputs(
        prompt_2, canvas_size=canvas_size
    )

    out_1, variables = _init_and_run_sft_diffusion(
        self.rng,
        prompt_1,
        x0_1,
        cm_1,
        cid_1,
        et_1,
        etm_1,
        prompt_len=prompt_len,
        canvas_size=canvas_size,
    )
    out_2, _ = _init_and_run_sft_diffusion(
        self.rng,
        prompt_2,
        x0_2,
        cm_2,
        cid_2,
        et_2,
        etm_2,
        prompt_len=prompt_len,
        canvas_size=canvas_size,
        variables=variables,
    )

    den_1 = out_1['output']['logits']
    den_2 = out_2['output']['logits']

    diff = jnp.abs(den_1 - den_2).max()
    self.assertGreater(
        float(diff),
        1e-3,
        'SFTDiffusion: Denoiser logits should differ when prompt changes',
    )


class SFTInferenceFnTest(absltest.TestCase):
  """Tests for SFTInferenceFn (outside-Flax inference path)."""

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(42)

  def _make_inference_fn_and_conditioning(self, prompt_len=6, canvas_size=4):
    """Build an SFTInferenceFn and matching conditioning via sft_encode.

    Args:
      prompt_len: Fixed padded maximum prompt length.
      canvas_size: Number of tokens per canvas.

    Returns:
      (inference_fn, conditioning, variables, xt, time)
    """
    gemma_network = _make_gemma_network()
    total_canvas_len = canvas_size
    batch_size = 1

    prompt = jnp.array([[5, 10, 15, 0, 0, 0]], dtype=jnp.int32)
    x0 = jnp.array([[7, 8, 9, 11]], dtype=jnp.int32)
    canvas_mask = jnp.array([[True, True, True, True]], dtype=jnp.bool_)
    selected_canvas_idx = jnp.zeros((batch_size,), dtype=jnp.int32)
    time = jnp.ones((batch_size, total_canvas_len, 1), dtype=jnp.float32)
    xt = jnp.array([[[7], [8], [9], [11]]], dtype=jnp.int32)

    # Wrapper that calls sft_encode + sft_decode to init ALL params
    # (including self_conditioner which is only used by __call__/decoder).
    class _FullWrapper(nn.Module):
      gemma_network: hd_gemma_network.WrappedDiffusionGemmaNetwork

      @nn.compact
      def __call__(
          self,
          prompt,
          x0,
          canvas_mask,
          selected_canvas_idx,
          xt,
          time,
          prompt_len,
          total_canvas_len,
          canvas_size,
      ):
        encoder_logits, kv_cache, positions, prompt_mask = _sft_encode(
            gemma_network=self.gemma_network,
            prompt=prompt,
            x0_tokens=x0,
            canvas_mask=canvas_mask,
            selected_canvas_idx=selected_canvas_idx,
            prompt_len=prompt_len,
            total_canvas_len=total_canvas_len,
            canvas_size=canvas_size,
        )
        denoiser_output = sft_model.sft_decode(
            gemma_network=self.gemma_network,
            xt=xt,
            time=time,
            kv_cache=kv_cache,
            positions=positions,
            prompt_mask=prompt_mask,
            canvas_mask=canvas_mask,
            selected_canvas_idx=selected_canvas_idx,
            prompt_len=prompt_len,
            total_canvas_len=total_canvas_len,
            canvas_size=canvas_size,
            is_training=False,
        )
        return {
            'encoder_logits': encoder_logits,
            'kv_cache': kv_cache,
            'positions': positions,
            'prompt_mask': prompt_mask,
            'logits': denoiser_output['logits'],
        }

    wrapper = _FullWrapper(gemma_network=gemma_network)
    variables = wrapper.init(
        self.rng,
        prompt=prompt,
        x0=x0,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        xt=xt,
        time=time,
        prompt_len=prompt_len,
        total_canvas_len=total_canvas_len,
        canvas_size=canvas_size,
    )
    full_out = wrapper.apply(
        variables,
        prompt=prompt,
        x0=x0,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        xt=xt,
        time=time,
        prompt_len=prompt_len,
        total_canvas_len=total_canvas_len,
        canvas_size=canvas_size,
    )

    # Build decoder conditioning from encoder outputs.
    prompt_mask = full_out['prompt_mask']
    positions = full_out['positions']
    kv_cache = full_out['kv_cache']
    full_seq_len = prompt_len + total_canvas_len
    cache_len = full_seq_len
    kv_positions = jnp.arange(cache_len)

    # Prompt attention.
    prompt_region = kv_positions < prompt_len
    prompt_pad_mask = jnp.zeros((batch_size, cache_len), dtype=jnp.bool_)
    prompt_pad_mask = prompt_pad_mask.at[:, :prompt_len].set(prompt_mask)
    prompt_attention = (
        prompt_region[None, None, :] & prompt_pad_mask[:, None, :]
    )

    # Canvas attention (single canvas, attend to canvas <= selected).
    in_canvas_region = (kv_positions >= prompt_len) & (
        kv_positions < prompt_len + total_canvas_len
    )
    kv_canvas_id = (kv_positions - prompt_len) // canvas_size
    canvas_attention = (
        kv_canvas_id[None, None, :] <= selected_canvas_idx[:, None, None]
    ) & in_canvas_region[None, None, :]
    canvas_valid_mask = jnp.zeros((batch_size, cache_len), dtype=jnp.bool_)
    canvas_valid_mask = canvas_valid_mask.at[
        :, prompt_len : prompt_len + total_canvas_len
    ].set(canvas_mask)
    canvas_attention = canvas_attention & canvas_valid_mask[:, None, :]
    attn_mask = prompt_attention | canvas_attention
    # Explicitly broadcast to [B, TotalCanvasLen, CacheLen].
    attn_mask = jnp.broadcast_to(
        attn_mask, (batch_size, total_canvas_len, cache_len)
    )

    canvas_positions = positions[:, prompt_len:]
    vocab_size = gemma_network.num_embed
    sc_logits = jnp.zeros(
        (batch_size, total_canvas_len, vocab_size), dtype=jnp.bfloat16
    )

    conditioning = {
        'kv_cache': kv_cache,
        'positions': canvas_positions,
        'attention_mask': attn_mask,
        'sc_logits': sc_logits,
    }

    # Extract gemma_network params from the wrapper's variables.
    gemma_network_params = variables['params']['gemma_network']

    inference_fn = sft_model.SFTInferenceFn(
        gemma_network=gemma_network,
        params=gemma_network_params,
    )
    return inference_fn, conditioning, variables, xt, time

  def test_inference_fn_produces_logits(self):
    """SFTInferenceFn should produce logits with the correct shape."""
    inference_fn, conditioning, _, xt, time = (
        self._make_inference_fn_and_conditioning()
    )

    output = inference_fn(time=time, xt=xt, conditioning=conditioning)

    self.assertIn('logits', output)
    # Shape should be [B, TotalCanvasLen, VocabSize].
    self.assertEqual(output['logits'].shape[0], 1)  # batch
    self.assertEqual(output['logits'].shape[1], 4)  # canvas_len

  def test_inference_fn_matches_bound_path(self):
    """SFTInferenceFn (unbound .apply) should match the bound sft_decode path.

    This verifies that calling the network via .apply() with explicit params
    produces the same result as calling it inside a Flax module.
    """
    prompt_len = 6
    canvas_size = 4

    prompt = jnp.array([[5, 10, 15, 0, 0, 0]], dtype=jnp.int32)
    x0 = jnp.array([[7, 8, 9, 11]], dtype=jnp.int32)
    canvas_mask = jnp.array([[True, True, True, True]], dtype=jnp.bool_)
    xt = jnp.array([[[7], [8], [9], [11]]], dtype=jnp.int32)

    # Run through the bound path (sft_encode + sft_decode inside Flax).
    bound_out, _ = _init_and_run(
        self.rng,
        prompt,
        x0,
        canvas_mask,
        xt,
        prompt_len=prompt_len,
        canvas_size=canvas_size,
    )

    # Now run through SFTInferenceFn (unbound .apply path).
    inference_fn, conditioning, _, _, time = (
        self._make_inference_fn_and_conditioning(
            prompt_len=prompt_len,
            canvas_size=canvas_size,
        )
    )

    unbound_out = inference_fn(time=time, xt=xt, conditioning=conditioning)

    np.testing.assert_allclose(
        bound_out['logits'],
        unbound_out['logits'],
        atol=1e-5,
        err_msg=(
            'SFTInferenceFn (unbound .apply) should match the bound'
            ' sft_decode path'
        ),
    )

  def test_inference_fn_sc_logits_squeeze(self):
    """SFTInferenceFn should squeeze sc_logits with trailing dim of 1."""
    inference_fn, conditioning, _, xt, time = (
        self._make_inference_fn_and_conditioning()
    )

    # Add an extra dim to sc_logits: [B, L, 1, V] — should be squeezed.
    sc_logits_4d = jnp.expand_dims(conditioning['sc_logits'], axis=-2)
    conditioning_4d = {**conditioning, 'sc_logits': sc_logits_4d}

    output_3d = inference_fn(time=time, xt=xt, conditioning=conditioning)
    output_4d = inference_fn(time=time, xt=xt, conditioning=conditioning_4d)

    np.testing.assert_allclose(
        output_3d['logits'],
        output_4d['logits'],
        atol=1e-6,
        err_msg=(
            'SFTInferenceFn should handle 4D sc_logits by squeezing the'
            ' trailing dim'
        ),
    )

  def test_inference_fn_sensitive_to_cache(self):
    """SFTInferenceFn logits should change when the KV cache changes."""
    inference_fn, conditioning, _, xt, time = (
        self._make_inference_fn_and_conditioning()
    )

    # Perturb the KV cache values.
    perturbed_cache = jax.tree.map(
        lambda x: x + 0.1
        if x.dtype == jnp.float32 or x.dtype == jnp.bfloat16
        else x,
        conditioning['kv_cache'],
    )
    conditioning_perturbed = {**conditioning, 'kv_cache': perturbed_cache}

    output_original = inference_fn(time=time, xt=xt, conditioning=conditioning)
    output_perturbed = inference_fn(
        time=time, xt=xt, conditioning=conditioning_perturbed
    )

    diff = jnp.abs(output_original['logits'] - output_perturbed['logits']).max()
    self.assertGreater(
        float(diff),
        1e-3,
        'Logits should differ when KV cache is perturbed',
    )


class MultiCanvasEncoderCacheTest(absltest.TestCase):
  """Tests denoising canvas 2 looks at encoder KV-cache from canvas 1."""

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(42)

  def test_decoder_logits_change_when_canvas1_tokens_change(self):
    """Decoder logits change when canvas 1 tokens change.

    When denoising canvas 2 (selected_canvas_idx=1), changing x0 tokens in
    canvas 1 should change the encoder KV-cache that the decoder uses, which
    in turn changes decoder logits on canvas 2.

    Setup: 2 canvases of size 4, total_canvas_len=8.
    Run 1: canvas 1 tokens = [1, 2, 3, 4]
    Run 2: canvas 1 tokens = [5, 6, 7, 8]
    Canvas 2 tokens and xt are identical in both runs.
    """
    canvas_size = 4
    num_canvases = 2
    total_canvas_len = canvas_size * num_canvases
    prompt_len = 4

    prompt = jnp.array([[10, 11, 0, 0]], dtype=jnp.int32)
    canvas_mask = jnp.ones((1, total_canvas_len), dtype=jnp.bool_)
    # selected_canvas_idx = 1 means we denoise canvas 2
    selected_canvas_idx = jnp.array([1], dtype=jnp.int32)
    # xt for denoiser — same in both runs
    xt = jnp.ones((1, total_canvas_len, 1), dtype=jnp.int32) * 3

    # Run 1: canvas 1 has tokens [1,2,3,4], canvas 2 has [20,21,22,23]
    x0_run1 = jnp.array([[1, 2, 3, 4, 20, 21, 22, 23]], dtype=jnp.int32)

    # Run 2: canvas 1 has tokens [5,6,7,8], canvas 2 has [20,21,22,23]
    x0_run2 = jnp.array([[5, 6, 7, 8, 20, 21, 22, 23]], dtype=jnp.int32)

    out1, variables = _init_and_run(
        self.rng,
        prompt,
        x0_run1,
        canvas_mask,
        xt,
        prompt_len=prompt_len,
        canvas_size=canvas_size,
        selected_canvas_idx=selected_canvas_idx,
    )
    out2, _ = _init_and_run(
        self.rng,
        prompt,
        x0_run2,
        canvas_mask,
        xt,
        prompt_len=prompt_len,
        canvas_size=canvas_size,
        selected_canvas_idx=selected_canvas_idx,
        variables=variables,
    )

    # Decoder logits should differ because canvas 1 tokens changed and the
    # encoder KV-cache that the denoiser of canvas 2 attends to is different.
    logits_diff = jnp.abs(out1['logits'] - out2['logits']).max()
    self.assertGreater(
        float(logits_diff),
        1e-3,
        'Decoder logits on canvas 2 should change when canvas 1 x0 tokens'
        ' change (the encoder KV-cache for canvas 1 should be visible).',
    )


class MultiCanvasEncoderCacheIsolationTest(absltest.TestCase):
  """Tests that encoder KV-cache for canvas 2 doesn't see canvas 2's own x0."""

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(42)

  def test_encoder_kv_cache_unchanged_when_canvas2_x0_changes(self):
    """Encoder KV-cache is unchanged when canvas 2 x0 changes.

    When denoising canvas 2 (selected_canvas_idx=1), changing the x0 tokens
    in canvas 2 should NOT change the encoder KV-cache entries that are
    actually used by the denoiser of canvas 2, because the encoder end_index
    is set to prompt_len + selected_canvas_idx * canvas_size, which excludes
    canvas 2.

    The KV-cache entries at indices [0, end_index) should be identical when
    only canvas 2's x0 tokens change.
    """
    canvas_size = 4
    num_canvases = 2
    total_canvas_len = canvas_size * num_canvases
    prompt_len = 4

    prompt = jnp.array([[10, 11, 0, 0]], dtype=jnp.int32)
    canvas_mask = jnp.ones((1, total_canvas_len), dtype=jnp.bool_)
    selected_canvas_idx = jnp.array([1], dtype=jnp.int32)
    xt = jnp.ones((1, total_canvas_len, 1), dtype=jnp.int32) * 3

    # Run 1: canvas 2 has tokens [20,21,22,23]
    x0_run1 = jnp.array([[1, 2, 3, 4, 20, 21, 22, 23]], dtype=jnp.int32)

    # Run 2: canvas 2 has tokens [30,31,5,6] — different from run 1
    x0_run2 = jnp.array([[1, 2, 3, 4, 30, 31, 5, 6]], dtype=jnp.int32)

    out1, variables = _init_and_run(
        self.rng,
        prompt,
        x0_run1,
        canvas_mask,
        xt,
        prompt_len=prompt_len,
        canvas_size=canvas_size,
        selected_canvas_idx=selected_canvas_idx,
    )
    out2, _ = _init_and_run(
        self.rng,
        prompt,
        x0_run2,
        canvas_mask,
        xt,
        prompt_len=prompt_len,
        canvas_size=canvas_size,
        selected_canvas_idx=selected_canvas_idx,
        variables=variables,
    )

    # The end_index = prompt_len + 1 * canvas_size = 4 + 4 = 8
    # So the denoiser uses KV-cache entries at indices [0, 8) which covers
    # the prompt (4) and canvas 1 (4). Canvas 2 starts at index 8.
    # Since canvas 1 and prompt are identical, those cache entries should match.
    end_index = prompt_len + 1 * canvas_size  # 8
    kv1 = out1['kv_cache']
    kv2 = out2['kv_cache']

    for layer_name in kv1:
      for key in ('k', 'v'):
        if key not in kv1[layer_name]:
          continue
        cache_slice_1 = kv1[layer_name][key][:, :end_index, ...]
        cache_slice_2 = kv2[layer_name][key][:, :end_index, ...]
        np.testing.assert_allclose(
            cache_slice_1,
            cache_slice_2,
            atol=1e-5,
            err_msg=(
                f'Encoder KV-cache layer {layer_name}/{key} at indices'
                f' [0, {end_index}) should not change when only canvas 2 x0'
                ' tokens are modified.'
            ),
        )


class GradientFlowTest(absltest.TestCase):
  """Tests that encoder and decoder losses produce non-zero gradients.

  Gradients must flow from the loss back to the model parameters. These tests
  catch cases where logits are accidentally detached from the computation graph.

  LoRA-specific gradient tests live in sft_model_lora_gradient_test.py
  (separate process to avoid Flax ModuleInterceptor state pollution).
  """

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(42)

  def test_encoder_logits_gradient_flows_to_params(self):
    """Gradient of CE(encoder_logits, target) w.r.t. params is non-zero."""
    gemma_network = _make_gemma_network()
    prompt_len = 6
    canvas_size = 4
    total_canvas_len = canvas_size

    prompt = jnp.array([[5, 10, 15, 0, 0, 0]], dtype=jnp.int32)
    x0 = jnp.array([[7, 8, 9, 11]], dtype=jnp.int32)
    canvas_mask = jnp.ones((1, total_canvas_len), dtype=jnp.bool_)
    selected_canvas_idx = jnp.zeros((1,), dtype=jnp.int32)

    # Target for AR loss: shifted x0 tokens.
    full_seq_len = prompt_len + total_canvas_len
    encoder_target = jnp.concatenate(
        [prompt[:, 1:], x0, jnp.zeros((1, 1), dtype=jnp.int32)], axis=1
    )[:, :full_seq_len]
    encoder_target_mask = jnp.ones((1, full_seq_len), dtype=jnp.float32)

    class _EncoderWrapper(nn.Module):
      gemma_network: hd_gemma_network.WrappedDiffusionGemmaNetwork

      @nn.compact
      def __call__(self, prompt, x0, canvas_mask, selected_canvas_idx):
        encoder_logits, _, _, _ = _sft_encode(
            gemma_network=self.gemma_network,
            prompt=prompt,
            x0_tokens=x0,
            canvas_mask=canvas_mask,
            selected_canvas_idx=selected_canvas_idx,
            prompt_len=prompt_len,
            total_canvas_len=total_canvas_len,
            canvas_size=canvas_size,
        )
        return encoder_logits

    wrapper = _EncoderWrapper(gemma_network=gemma_network)
    variables = wrapper.init(
        self.rng,
        prompt=prompt,
        x0=x0,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
    )

    def loss_fn(params):
      encoder_logits = wrapper.apply(
          {'params': params},
          prompt=prompt,
          x0=x0,
          canvas_mask=canvas_mask,
          selected_canvas_idx=selected_canvas_idx,
      )
      # Cross-entropy loss (same as EncoderARLoss).

      loss = optax.softmax_cross_entropy_with_integer_labels(
          encoder_logits, encoder_target
      )
      masked_loss = loss * encoder_target_mask
      return jnp.sum(masked_loss) / jnp.maximum(
          jnp.sum(encoder_target_mask), 1.0
      )

    grads = jax.grad(loss_fn)(variables['params'])

    # At least some gradients should be non-zero.
    grad_norm = jnp.sqrt(
        sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads))
    )
    self.assertGreater(
        float(grad_norm),
        1e-6,
        'Encoder loss should produce non-zero gradients w.r.t. model params.',
    )

  def test_decoder_logits_gradient_flows_to_params(self):
    """Gradient of CE(decoder_logits, target) w.r.t. params is non-zero."""
    gemma_network = _make_gemma_network()
    prompt_len = 6
    canvas_size = 4
    total_canvas_len = canvas_size

    prompt = jnp.array([[5, 10, 15, 0, 0, 0]], dtype=jnp.int32)
    x0 = jnp.array([[7, 8, 9, 11]], dtype=jnp.int32)
    canvas_mask = jnp.ones((1, total_canvas_len), dtype=jnp.bool_)
    selected_canvas_idx = jnp.zeros((1,), dtype=jnp.int32)
    xt = jnp.array([[[7], [8], [9], [11]]], dtype=jnp.int32)
    time = jnp.ones((1, total_canvas_len, 1), dtype=jnp.float32)

    # Target for decoder loss: the clean x0 tokens.
    decoder_target = x0  # [B, TotalCanvasLen]

    class _DecoderWrapper(nn.Module):
      gemma_network: hd_gemma_network.WrappedDiffusionGemmaNetwork

      @nn.compact
      def __call__(
          self, prompt, x0, canvas_mask, selected_canvas_idx, xt, time
      ):
        _, kv_cache, positions, prompt_mask = _sft_encode(
            gemma_network=self.gemma_network,
            prompt=prompt,
            x0_tokens=x0,
            canvas_mask=canvas_mask,
            selected_canvas_idx=selected_canvas_idx,
            prompt_len=prompt_len,
            total_canvas_len=total_canvas_len,
            canvas_size=canvas_size,
        )
        denoiser_output = sft_model.sft_decode(
            gemma_network=self.gemma_network,
            xt=xt,
            time=time,
            kv_cache=kv_cache,
            positions=positions,
            prompt_mask=prompt_mask,
            canvas_mask=canvas_mask,
            selected_canvas_idx=selected_canvas_idx,
            prompt_len=prompt_len,
            total_canvas_len=total_canvas_len,
            canvas_size=canvas_size,
            is_training=False,
        )
        return denoiser_output['logits']

    wrapper = _DecoderWrapper(gemma_network=gemma_network)
    variables = wrapper.init(
        self.rng,
        prompt=prompt,
        x0=x0,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        xt=xt,
        time=time,
    )

    def loss_fn(params):
      decoder_logits = wrapper.apply(
          {'params': params},
          prompt=prompt,
          x0=x0,
          canvas_mask=canvas_mask,
          selected_canvas_idx=selected_canvas_idx,
          xt=xt,
          time=time,
      )

      loss = optax.softmax_cross_entropy_with_integer_labels(
          decoder_logits, decoder_target
      )
      return jnp.mean(loss)

    grads = jax.grad(loss_fn)(variables['params'])

    grad_norm = jnp.sqrt(
        sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads))
    )
    self.assertGreater(
        float(grad_norm),
        1e-6,
        'Decoder loss should produce non-zero gradients w.r.t. model params.',
    )


class EncoderARLossTest(absltest.TestCase):

  def test_encoder_ar_loss(self):
    loss_fn = sft_model.EncoderARLoss()

    # Shape: [B=2, SeqLen=3, VocabSize=4]
    # We use large values to make softmax close to one-hot
    encoder_logits = jnp.array([
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
        ],
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
        ],
    ])

    # Shape: [B=2, SeqLen=3]
    # First example: perfect prediction for masked elements
    # Second example: wrong prediction for masked elements
    encoder_target = jnp.array([
        [0, 1, 3],  # Perfect for 0 and 1, wrong for 2 (but 2 is masked)
        [1, 0, 2],  # Wrong for 0 and 1, perfect for 2 (but 2 is masked)
    ])

    # Shape: [B=2, SeqLen=3]
    encoder_target_mask = jnp.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    loss = loss_fn.get_values(
        encoder_logits=encoder_logits,
        encoder_target=encoder_target,
        encoder_target_mask=encoder_target_mask,
    )

    # Expected loss for first example: approx 0
    # (perfect predictions where mask=1)
    # Expected loss for second example: approx 10
    # (wrong predictions where mask=1)
    # Softmax of [10, 0, 0, 0] is approx [1, 0, 0, 0]
    # Cross entropy with label 1 (wrong) is -log(approx 0) = approx 10
    # Cross entropy with label 0 (correct) is -log(approx 1) = approx 0

    np.testing.assert_allclose(loss[0], 0.0, atol=1e-3)
    # loss[1] should be approx 10.0
    self.assertGreater(loss[1], 5.0)


if __name__ == '__main__':
  absltest.main()
