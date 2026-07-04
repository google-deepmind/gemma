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

"""Diffusion sampler."""

import dataclasses
import functools
from typing import cast, override

import flax.struct
from gemma.diffusion import _early_stopping
from gemma.diffusion import _transformer
from gemma.gm.nn.gemma4 import _config
from gemma.gm.text import _sampler_loop
from gemma.gm.typing import _common
import jax
import jax.numpy as jnp
from kauldron.ktyping import Bool, Float, Int, PRNGKey, typechecked  # pylint: disable=g-multiple-import,g-importing-member

# Minimum value for the temperature to ensure numerical stability.
_MIN_TEMP = 1e-12
PAD_TOKEN = 0


Embeddings = Float['*B L D']
Logits = Float['*B L V']
NoiseProportion = Float['*B']
Tokens = Int['*B L']


@dataclasses.dataclass(frozen=True)
class LinearSchedule:
  """Linear noise schedule."""

  def noise_probability(self, noise_proportion: Float) -> Float:  # pyrefly: ignore[not-a-type]
    return noise_proportion

  def derivative_noise_probability(self, noise_proportion: Float) -> Float:  # pyrefly: ignore[not-a-type]
    del noise_proportion
    return jnp.array(1.0)


@dataclasses.dataclass(frozen=True)
class DiffusionProcess:
  """Diffusion process for multinomial diffusion."""

  noise_schedule: LinearSchedule = dataclasses.field(
      default_factory=LinearSchedule
  )

  def get_initial_sample(
      self,
      rng: PRNGKey,
      batch_size: int,
      canvas_length: int,
      text_vocab_size: int,
  ) -> Tokens:  # pyrefly: ignore[not-a-type]
    """Create an initial noisy canvas of random tokens for sampling."""

    return jax.random.randint(
        rng,
        shape=(batch_size, canvas_length),
        minval=0,
        maxval=text_vocab_size,
    )

  def add_noise_to_tokens(
      self,
      rng: PRNGKey,
      canvas_tokens: Tokens,  # pyrefly: ignore[not-a-type]
      noise_proportion: Float['*B'],  # pyrefly: ignore[not-a-type]
      text_vocab_size: int,
  ) -> Tokens:  # pyrefly: ignore[not-a-type]
    """Adds noise to the tokens."""
    rng_mask, rng_tokens = jax.random.split(rng)

    prob_noise = jax.vmap(self.noise_schedule.noise_probability)(
        noise_proportion
    )
    noise_mask = jax.random.bernoulli(
        rng_mask,
        p=prob_noise[:, None],
        shape=canvas_tokens.shape,
    )
    random_tokens = jax.random.randint(
        rng_tokens,
        shape=canvas_tokens.shape,
        minval=0,
        maxval=text_vocab_size,
    )
    return jnp.where(noise_mask, random_tokens, canvas_tokens)


@dataclasses.dataclass(frozen=True)
class SampleFromPredictions:
  """Samples tokens from the predicted logits.

  Selects tokens based on the model's confidence and renoises non-selected
  positions.

  Attributes:
    entropy_bound: Confidence threshold controlling how many tokens are accepted
      per step. Lower values accept fewer tokens (more conservative).
    text_vocab_size: Vocabulary size, needed for renoising non-selected tokens.
  """

  entropy_bound: float = 0.1
  text_vocab_size: int = 0

  def __call__(
      self,
      *,
      rng: PRNGKey,
      denoiser_logits: Logits,  # pyrefly: ignore[not-a-type]
      canvas: Tokens,  # pyrefly: ignore[not-a-type]
      current_noise_proportion: NoiseProportion,  # pyrefly: ignore[not-a-type]
      target_noise_proportion: NoiseProportion,  # pyrefly: ignore[not-a-type]
  ) -> Tokens:  # pyrefly: ignore[not-a-type]
    """Returns the sample step output.

    Args:
      rng: RNG key.
      denoiser_logits: Shaped logits from the denoiser.
      canvas: The current noisy canvas from the previous step.
      current_noise_proportion: The noise level of the current canvas.
      target_noise_proportion: The desired noise level after this step.

    Returns:
      The denoised tokens after applying confidence-based selection and
      renoising non-selected positions.
    """
    del current_noise_proportion, target_noise_proportion

    categorical_rng, noise_rng = jax.random.split(rng)
    denoiser_tokens = jax.random.categorical(
        categorical_rng, denoiser_logits.astype(jnp.float32)
    )
    batch_size = canvas.shape[0]

    # Compute per-token entropy from the logits.
    log_probs = jax.nn.log_softmax(denoiser_logits.astype(jnp.float32))
    probs = jnp.exp(log_probs)
    safe_log_probs = jnp.where(probs == 0, 0.0, log_probs)
    token_entropy = -jnp.sum(safe_log_probs * probs, axis=-1)  # [B, L]

    # Sort tokens by entropy (ascending) and build the selection mask.
    sorted_index = jnp.argsort(token_entropy, axis=-1)
    sorted_entropy = jnp.take_along_axis(token_entropy, sorted_index, axis=-1)
    accumulated_entropy = jnp.cumsum(sorted_entropy, axis=-1)

    # Accept k tokens where accumulated - sorted <= entropy_bound.
    sorted_selection_mask = (
        accumulated_entropy - sorted_entropy
    ) <= self.entropy_bound

    # Scatter the sorted mask back to original positions.
    selection_mask = (
        jnp.zeros_like(sorted_index, dtype=jnp.bool_)
        .at[jnp.arange(batch_size)[:, None], sorted_index]
        .set(sorted_selection_mask)
    )

    # Renoise all non-selected tokens with uniform random tokens.
    # Selected positions get denoiser tokens.
    random_tokens = jax.random.randint(
        noise_rng,
        shape=canvas.shape,
        minval=0,
        maxval=self.text_vocab_size,
    )
    output_tokens = jnp.where(selection_mask, denoiser_tokens, random_tokens)

    return output_tokens


@flax.struct.dataclass
class SampleStepOutput:
  """Output of the diffusion sampler.

  Attributes:
    sampled_tokens: The tokens sampled in this step.
    sc_embeddings: The self conditioning signal to feed back into the
      transformer.
    logits: The predicted logits from this step.
    modified_tokens_mask: A mask indicating which tokens were modified during
      this sampling step.
  """

  sampled_tokens: Tokens  # pyrefly: ignore[not-a-type]
  sc_embeddings: Embeddings  # pyrefly: ignore[not-a-type]
  logits: Logits  # pyrefly: ignore[not-a-type]
  modified_tokens_mask: Bool['*B L']  # pyrefly: ignore[not-a-type]


@flax.struct.dataclass
class _WhileLoopCarry:
  """Carry state for the jax.lax.while_loop in sample_next_canvas."""

  step: Int['']  # pyrefly: ignore[not-a-type]
  canvas: Tokens  # pyrefly: ignore[not-a-type]
  sc_embeddings: Embeddings  # pyrefly: ignore[not-a-type]
  rng: PRNGKey
  done: Bool['B']  # pyrefly: ignore[not-a-type, unknown-name]


@dataclasses.dataclass(frozen=True, kw_only=True)
class AnnealingTemperatureShaperConfig:
  """Configuration for AnnealingTemperatureShaper.

  Attributes:
    exponent: Controls the shape of the temperature curve as a function of
      `noise_proportion`. The temperature interpolates from `max_temperature`
      (when `noise_proportion`=1) down to `min_temperature` (when
      `noise_proportion`=0) based on the formula: `factor = 1 - (1 -
      noise_proportion)**exponent`. - exponent = 1: Linear decrease in
      temperature. - exponent > 1: Temperature decreases slower initially,
      faster later. - exponent < 1: Temperature decreases faster initially,
      slower later.
    max_temperature: The temperature used at the beginning (noise_proportion=1).
    min_temperature: The temperature used at the end (noise_proportion=0).
  """

  exponent: float = 1.0
  max_temperature: float = 0.8
  min_temperature: float = 0.4

  def __post_init__(self):
    if self.min_temperature < _MIN_TEMP:
      raise ValueError(f'{self.min_temperature=} should be >= {_MIN_TEMP=}')
    if self.max_temperature < self.min_temperature:
      raise ValueError(
          f'{self.max_temperature=} should be >= {self.min_temperature=}'
      )

  def make(self) -> 'AnnealingTemperatureShaper':
    return AnnealingTemperatureShaper(config=self)


@dataclasses.dataclass(frozen=True)
class AnnealingTemperatureShaper:
  """Scales logits by a temperature that anneals based on noise_proportion.

  The temperature decreases from `max_temperature` (when noise_proportion=1)
  down to `min_temperature` (when noise_proportion=0) according to a power law
  controlled by the `exponent` parameter in the config.
  """

  config: AnnealingTemperatureShaperConfig

  @typechecked
  def __call__(
      self,
      logits: Float['*B L V'],  # pyrefly: ignore[not-a-type]
      noise_proportion: Float['*B'],  # pyrefly: ignore[not-a-type]
  ) -> Float['*B L V']:  # pyrefly: ignore[not-a-type]

    # Calculate temperature directly from noise_proportion.
    # noise_proportion goes from ~1 down to ~0.
    # (1 - noise_proportion) goes from ~0 up to ~1.
    # (1 - noise_proportion)**exponent goes from ~0 up to ~1.
    # 1 - (1 - noise_proportion)**exponent goes from ~1 down to ~0.
    # This matches the range needed for the final scaling.
    temperature_fraction = (
        1.0
        - (1.0 - noise_proportion.astype(logits.dtype)) ** self.config.exponent
    )

    # Scale to the final range [min_temperature, max_temperature].
    temperature = (
        temperature_fraction
        * (self.config.max_temperature - self.config.min_temperature)
    ) + self.config.min_temperature  # Shape [Batch]
    temperature = temperature.astype(logits.dtype)

    # Apply temperature scaling.
    out_logits = logits / temperature[:, None, None]

    return out_logits.astype(logits.dtype)


@typechecked
def _truncate_canvas_at_stop_tokens(
    canvas: Tokens,  # pyrefly: ignore[not-a-type]
    *,
    end_tokens: tuple[int, ...],
    canvas_length: int,
    done: Bool['B'],  # pyrefly: ignore[not-a-type, unknown-name]
) -> tuple[Tokens, Bool['B']]:  # pyrefly: ignore[not-a-type, unknown-name]
  """Replaces tokens after the first stop token with PAD_TOKEN."""
  end_tokens_arr = jnp.array(end_tokens, dtype=jnp.int32)
  is_stop_token = jnp.isin(canvas, end_tokens_arr)
  batch_has_stop_token = jnp.any(is_stop_token, axis=-1)

  first_stop_idx = jnp.argmax(is_stop_token, axis=-1)

  seq_idx = jnp.arange(canvas_length)[None, :]
  keep_mask = seq_idx <= jnp.where(
      batch_has_stop_token[:, None],
      first_stop_idx[:, None],
      canvas_length,
  )
  keep_mask = keep_mask & ~done[:, None]
  canvas = jnp.where(keep_mask, canvas, PAD_TOKEN)

  return canvas, batch_has_stop_token


@dataclasses.dataclass(frozen=True, kw_only=True)
class DiffusionSampler(_sampler_loop.SamplerLoop):
  """Diffusion sampler, combining the sampling loop and diffusion algorithm.

  On top of the base SamplerLoop, holds diffusion-specific attributes and
  overrides the `_sample_step` method to implement block-wise diffusion
  sampling. Each `_sample_step` produces a full canvas of tokens.
  """

  diffusion_process: DiffusionProcess = dataclasses.field(
      default_factory=DiffusionProcess
  )
  logit_shaper: AnnealingTemperatureShaper = dataclasses.field(
      default_factory=lambda: AnnealingTemperatureShaper(
          config=AnnealingTemperatureShaperConfig()
      )
  )
  sample_from_predictions: SampleFromPredictions = dataclasses.field(
      default_factory=SampleFromPredictions
  )
  canvas_length: int
  max_denoising_steps: int
  text_vocab_size: int
  sliding_window_size: int | None = None
  early_stop_fn: _early_stopping.EarlyStopFn = dataclasses.field(
      default_factory=_early_stopping.NoEarlyStop
  )

  @typechecked
  def sample_next_canvas(
      self,
      *,
      canvas_length: int,
      max_denoising_steps: int,
      batch_size: int,
      cache: _config.Cache | None,
      params: _common.Params,
      rng: PRNGKey,
      last_token_pos: Int['*B'] | None = None,
      full_attention_mask: Bool['*B CacheLength'] | None = None,
  ) -> Tokens:  # pyrefly: ignore[not-a-type]
    """Samples a complete denoised canvas from an initial noisy canvas.

    This function performs a multi-step denoising process, starting from a
    fully noisy canvas and iteratively refining it over `max_denoising_steps`.

    Args:
      canvas_length: The length of the token sequence to sample.
      max_denoising_steps: The number of denoising steps to perform.
      batch_size: The batch size.
      cache: Optional KV cache for the transformer.
      params: The transformer model parameters.
      rng: JAX PRNGKey.
      last_token_pos: Sequence-specific unpadded ending position of each prompt.
      full_attention_mask: full attention mask used for previous canvas

    Returns:
      The fully denoised token canvas of shape [*B, canvas_length].
    """
    initial_canvas_rng, step_rng = jax.random.split(rng)
    del rng

    if cache is not None:
      cache_layer = list(cache.values())[0]
      cache_length = cache_layer['k'].shape[1]
      samples_in_cache: Int['*B'] = cache_layer['end_index']  # pyrefly: ignore[not-a-type]
      if last_token_pos is not None:
        unpadded_valid_tokens = last_token_pos
        if hasattr(self.model, 'keep_last_prefill_kv') and self.model.keep_last_prefill_kv:
          unpadded_valid_tokens = unpadded_valid_tokens + 1
      else:
        unpadded_valid_tokens = samples_in_cache
      positions = unpadded_valid_tokens[:, None] + jnp.arange(canvas_length)[None, :]
    else:
      cache_length = None
      samples_in_cache = None
      unpadded_valid_tokens = None
      positions = jnp.broadcast_to(
          jnp.arange(canvas_length)[None, :], (batch_size, canvas_length)
      )

    attention_mask = _make_global_attention_mask(
        batch_size=batch_size,
        canvas_length=canvas_length,
        cache_length=cache_length,
        num_valid_tokens=unpadded_valid_tokens,
        physical_valid_tokens=samples_in_cache,
        full_attention_mask=full_attention_mask,
    )

    block_local_mask = _make_block_local_attention_mask(
        batch_size=batch_size,
        canvas_length=canvas_length,
        sliding_window_size=self.sliding_window_size,
        cache_length=cache_length,
        num_valid_tokens=unpadded_valid_tokens,
        physical_valid_tokens=samples_in_cache,
    )

    initial_tokens = self.diffusion_process.get_initial_sample(
        rng=initial_canvas_rng,
        batch_size=batch_size,
        canvas_length=canvas_length,
        text_vocab_size=self.text_vocab_size,
    )

    # Pre-compute noise proportions at each step boundary.
    # noise_proportions[i] = 1.0 - i / max_denoising_steps, so:
    #   noise_proportions[0] = 1.0 (fully noisy)
    #   noise_proportions[max_denoising_steps] = 0.0 (fully denoised)
    # At step i: current = noise_proportions[step],
    #            target  = noise_proportions[step + 1].
    noise_proportions = (
        1.0 - jnp.arange(max_denoising_steps + 1) / max_denoising_steps
    )

    embed_dim = cast(_config.TransformerConfig, self.model.config).embed_dim

    def cond_fn(carry: _WhileLoopCarry) -> Bool['']:  # pyrefly: ignore[not-a-type]
      return jnp.logical_and(
          ~jnp.all(carry.done),
          carry.step < max_denoising_steps,
      )

    def body_fn(carry: _WhileLoopCarry) -> _WhileLoopCarry:
      step = carry.step
      next_rng, sample_rng = jax.random.split(carry.rng)

      current_noise_proportion = jnp.full(
          (batch_size,), noise_proportions[step]
      )
      target_noise_proportion = jnp.full(
          (batch_size,), noise_proportions[step + 1]
      )
      out = self.sample_step(
          canvas=carry.canvas,
          sc_embeddings=carry.sc_embeddings,
          cache=cache,
          positions=positions,
          attention_mask=attention_mask,
          sliding_attention_mask=block_local_mask,
          current_noise_proportion=current_noise_proportion,
          target_noise_proportion=target_noise_proportion,
          params=params,
          rng=sample_rng,
      )

      new_done = jnp.logical_or(
          carry.done,
          self.early_stop_fn.should_stop(
              step=step,
              canvas=out.sampled_tokens,
              previous_canvas=carry.canvas,
              logits=out.logits,
          ),
      )

      # Freeze canvas for done elements. sc_embeddings don't need freezing
      # because done elements' canvases are frozen, so model outputs for
      # them are discarded on the next iteration anyway.
      canvas = jnp.where(carry.done[:, None], carry.canvas, out.sampled_tokens)

      return _WhileLoopCarry(
          step=step + 1,
          canvas=canvas,
          sc_embeddings=out.sc_embeddings.astype(carry.sc_embeddings.dtype),
          rng=next_rng,
          done=new_done,
      )

    init_carry = _WhileLoopCarry(
        step=jnp.int32(0),
        canvas=initial_tokens,
        sc_embeddings=jnp.zeros(
            (batch_size, canvas_length, embed_dim),
            dtype=jnp.bfloat16,
        ),
        rng=step_rng,
        done=jnp.zeros(batch_size, dtype=jnp.bool_),
    )

    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)

    return final_carry.canvas

  @functools.partial(jax.jit, static_argnames=('self',))
  @typechecked
  @override
  def _sample_step(
      self,
      state: _sampler_loop.SamplingState,
      *,
      params: _common.Params,
  ) -> _sampler_loop.SamplingState:
    """Single diffusion sampling step (full canvas, multiple tokens)."""
    next_rng, sample_rng = jax.random.split(state.rng)

    cache = state.cache
    cache_layer = list(cache.values())[0]
    batch_size = cache_layer['end_index'].shape[0]

    canvas = self.sample_next_canvas(
        canvas_length=self.canvas_length,
        max_denoising_steps=self.max_denoising_steps,
        batch_size=batch_size,
        cache=cache,
        params=params,
        rng=sample_rng,
        last_token_pos=state.last_token_pos,
        full_attention_mask=state.full_attention_mask,
    )

    canvas, batch_has_stop_token = _truncate_canvas_at_stop_tokens(
        canvas,
        end_tokens=self.end_tokens,
        canvas_length=self.canvas_length,
        done=state.done,
    )

    cache = self.append_tokens_to_cache(
        tokens=canvas,
        cache=cache,
        params=params,
        last_token_pos=state.last_token_pos,
    )

    done = state.done | batch_has_stop_token

    indices = jnp.arange(self.canvas_length) + state.step
    predicted_tokens = state.predicted_tokens.at[:, indices].set(canvas)

    return _sampler_loop.SamplingState(
        step=state.step + self.canvas_length,
        done=done,
        last_token=canvas[:, -1],
        last_token_pos=state.last_token_pos + self.canvas_length,
        predicted_tokens=predicted_tokens,
        cache=cache,
        rng=next_rng,
        init_cache_length=state.init_cache_length,
        full_attention_mask=state.full_attention_mask,
    )

  @typechecked
  def sample_step(
      self,
      *,
      canvas: Tokens,  # pyrefly: ignore[not-a-type]
      sc_embeddings: Embeddings,  # pyrefly: ignore[not-a-type]
      cache: _config.Cache | None,
      positions: Int['*B L'] | None,
      attention_mask: Bool['*B CanvasLength CachePlusCanvasLength'] | None,
      sliding_attention_mask: (
          Bool['*B CanvasLength CachePlusCanvasLength'] | None
      ) = None,
      current_noise_proportion: NoiseProportion,  # pyrefly: ignore[not-a-type]
      target_noise_proportion: NoiseProportion,  # pyrefly: ignore[not-a-type]
      params: _common.Params,
      rng: PRNGKey,
  ) -> SampleStepOutput:
    """Performs a single sampling step."""

    transformer_output = self.model.apply(
        {'params': params},
        tokens=canvas,
        sc_embeddings=sc_embeddings,  # pyrefly: ignore[unexpected-keyword]
        cache=cache,
        positions=positions,
        attention_mask=attention_mask,
        sliding_attention_mask=sliding_attention_mask,  # pyrefly: ignore[unexpected-keyword]
        method=_transformer.DiffusionMixin.call_with_self_conditioning,  # pyrefly: ignore[unexpected-keyword]
    )

    shaped_prediction = self.logit_shaper(
        logits=transformer_output.logits,  # pyrefly: ignore[missing-attribute]
        noise_proportion=current_noise_proportion,
    )

    sampled = self.sample_from_predictions(
        rng=rng,
        denoiser_logits=shaped_prediction,
        canvas=canvas,
        current_noise_proportion=current_noise_proportion,
        target_noise_proportion=target_noise_proportion,
    )

    # Encode the shaped logits into embeddings for self-conditioning in the
    # next denoising step, using the model's own Embedder.encode_logits method.
    new_sc_embeddings = self.model.apply(
        {'params': params},
        shaped_prediction,
        method=lambda self, x: self.embedder.encode_logits(x),  # pyrefly: ignore[unexpected-keyword]
    )

    return SampleStepOutput(
        sc_embeddings=new_sc_embeddings,
        logits=shaped_prediction,
        sampled_tokens=sampled,
        modified_tokens_mask=sampled != canvas,
    )

  @typechecked
  def append_tokens_to_cache(
      self,
      *,
      tokens: Tokens,  # pyrefly: ignore[not-a-type]
      cache: _config.Cache,
      params: _common.Params,
      last_token_pos: Int['*B'] | None = None,
  ) -> _config.Cache:
    """Inserts tokens into the cache via a transformer forward pass.

    Uses a causal attention mask so that each token can attend to all valid
    cached tokens and to preceding tokens in the input, but not to future
    tokens.

    Args:
      tokens: Tokens to insert, shaped [batch_size, seq_len].
      cache: The current KV cache.
      params: Model parameters.
      last_token_pos: Sequence-specific unpadded ending position of each prompt.

    Returns:
      The updated cache with the tokens inserted.
    """
    return self._insert_tokens(
        tokens=tokens,
        cache=cache,
        params=params,
        last_token_pos=last_token_pos,
    )

  def _insert_tokens(
      self,
      tokens: Int['B L'],
      cache: _config.Cache,
      params: _common.Params,
      last_token_pos: Int['*B'] | None = None,
  ) -> _config.Cache:
    seq_len = tokens.shape[1]

    cache_layer = list(cache.values())[0]
    cache_length = cache_layer['k'].shape[1]
    samples_in_cache: Int['B'] = cache_layer['end_index']  # pyrefly: ignore[not-a-type, unknown-name]
    if last_token_pos is not None:
      unpadded_valid_tokens = last_token_pos
      if hasattr(self.model, 'keep_last_prefill_kv') and self.model.keep_last_prefill_kv:
        unpadded_valid_tokens = unpadded_valid_tokens + 1
    else:
      unpadded_valid_tokens = samples_in_cache

    positions = unpadded_valid_tokens[:, None] + jnp.arange(seq_len)[None, :]

    attention_mask = _make_causal_attention_mask(
        batch_size=tokens.shape[0],
        canvas_length=seq_len,
        cache_length=cache_length,
        num_valid_cache_tokens=unpadded_valid_tokens,
        physical_valid_cache_tokens=samples_in_cache,
    )

    output = self.model.apply(
        {'params': params},
        tokens=tokens,
        cache=cache,
        positions=positions,
        attention_mask=attention_mask,
    )

    return output.cache  # pyrefly: ignore[missing-attribute]


@typechecked
def _make_global_attention_mask(
    batch_size: int,
    canvas_length: int,
    cache_length: int | None,
    num_valid_tokens: Int['*B'] | None,
    physical_valid_tokens: Int['*B'] | None = None,
    full_attention_mask: Bool['*B CacheLength'] | None = None,
) -> Bool['*B CanvasLength CacheLength']:  # pyrefly: ignore[not-a-type]
  """Create attention mask for the diffusion sampler.

  The canvas has full self attention.  The cache is left aligned, right padded,
  has 1's for valid samples and 0's for padding.

  The canvas is inserted into the cache before attention so the total mask
  length is just cache length.

  Args:
    batch_size: The batch size.
    canvas_length: The length of the canvas.
    cache_length: The length of the cache. If None, no cache is used.
    num_valid_tokens: The number of valid tokens in the cache. Required if
      cache_length is not None.
    physical_valid_tokens: The physical end index of the cache.
    full_attention_mask: The full attention mask for prompt and cache.

  Returns:
    The attention mask.
  """

  if cache_length is None:
    return jnp.ones((batch_size, canvas_length, canvas_length), dtype=jnp.bool_)

  if num_valid_tokens is None:
    raise ValueError(
        'num_valid_samples must be provided if cache_length is set.'
    )

  if physical_valid_tokens is None:
    physical_valid_tokens = num_valid_tokens

  prompt_mask = jnp.arange(cache_length)[None, :] < num_valid_tokens[:, None]
  if full_attention_mask is not None:
    prompt_mask = prompt_mask & full_attention_mask

  canvas_mask = (jnp.arange(cache_length)[None, :] >= physical_valid_tokens[:, None]) & (
      jnp.arange(cache_length)[None, :] < (physical_valid_tokens + canvas_length)[:, None]
  )

  mask = prompt_mask | canvas_mask

  return jnp.broadcast_to(
      mask[:, None, :], (batch_size, canvas_length, cache_length)
  )


@typechecked
def _make_causal_attention_mask(
    batch_size: int,
    canvas_length: int,
    cache_length: int | None,
    num_valid_cache_tokens: Int['B'] | None,  # pyrefly: ignore[unknown-name]
    physical_valid_cache_tokens: Int['B'] | None = None,
) -> Bool['B SeqLen CacheLength']:  # pyrefly: ignore[not-a-type]
  """Create a causal attention mask for inserting tokens into the cache.

  Args:
    batch_size: The batch size.
    canvas_length: Number of new tokens being inserted.
    cache_length: Total cache size.
    num_valid_cache_tokens: Per-batch number of samples in the cache before
      inserting new tokens.  If this is larger than cache_length the cache is
      assumed to be full and the oldest samples have been evicted.
    physical_valid_cache_tokens: The physical end index of the cache.

  Returns:
    Attention mask of shape [batch_size, canvas_length, cache_length].
  """

  if cache_length is None:
    causal_mask = jnp.tril(
        jnp.ones((canvas_length, canvas_length), dtype=jnp.bool_)
    )
    return jnp.broadcast_to(
        causal_mask[None, :, :], (batch_size, canvas_length, canvas_length)
    )

  if num_valid_cache_tokens is None:
    raise ValueError(
        'num_valid_cache_tokens must be provided if cache_length is set.'
    )

  if physical_valid_cache_tokens is None:
    physical_valid_cache_tokens = num_valid_cache_tokens

  valid_entries = jnp.minimum(num_valid_cache_tokens, cache_length)

  # 1. Fill base mask up to the number of valid tokens in the cache.
  mask = jnp.broadcast_to(
      jnp.arange(cache_length)[None, None, :] < valid_entries[:, None, None],
      (batch_size, canvas_length, cache_length),
  )

  # 2. Append a lower triangular matrix at the (wrapped) write positions.
  write_indices = (
      physical_valid_cache_tokens[:, None] + jnp.arange(canvas_length)[None, :]
  ) % cache_length

  batch_idx = jnp.arange(batch_size)[:, None, None]
  seq_idx = jnp.arange(canvas_length)[None, :, None]
  write_idx = write_indices[:, None, :]

  causal_mask = jnp.tril(
      jnp.ones((canvas_length, canvas_length), dtype=jnp.bool_)
  )

  mask = mask.at[batch_idx, seq_idx, write_idx].set(causal_mask[None, :, :])

  return mask


@typechecked
def _make_block_local_attention_mask(
    batch_size: int,
    canvas_length: int,
    sliding_window_size: int | None,
    cache_length: int | None,
    num_valid_tokens: Int['*B'] | None,
    physical_valid_tokens: Int['*B'] | None = None,
) -> Bool['*B CanvasLength CacheLength'] | None:
  """Create block-local attention mask for LOCAL_SLIDING layers in diffusion.

  Block-local attention semantics: all canvas tokens share
  the same context window and have full self-attention among themselves.

  For each canvas query token, the mask allows attending to:
    - Context tokens in [context_end - sliding_window_size, context_end),
      where context_end is the position of the first canvas token. This window
      is the same for ALL canvas tokens.
    - All other canvas tokens (full bidirectional self-attention).

  Args:
    batch_size: The batch size.
    canvas_length: The length of the canvas.
    sliding_window_size: The sliding window size. If None, returns None (global
      attention layers will use the regular attention_mask).
    cache_length: The length of the cache. If None, no cache is used.
    num_valid_tokens: The number of valid tokens in the cache before inserting
      canvas tokens. Required if cache_length is not None.
    physical_valid_tokens: The physical end index of the cache.

  Returns:
    The block-local attention mask, or None if sliding_window_size is None.
  """
  if sliding_window_size is None:
    return None

  if cache_length is None:
    # No cache = no context. Full canvas self-attention.
    return jnp.ones((batch_size, canvas_length, canvas_length), dtype=jnp.bool_)

  if num_valid_tokens is None:
    raise ValueError(
        'num_valid_tokens must be provided if cache_length is set.'
    )

  if physical_valid_tokens is None:
    physical_valid_tokens = num_valid_tokens

  # Context boundary: first canvas position in the cache.
  # context_end = num_valid_tokens (index of first canvas token)
  context_end = num_valid_tokens  # [B]
  context_start = jnp.maximum(context_end - sliding_window_size, 0)  # [B]

  cache_indices = jnp.arange(cache_length)[None, :]  # [1, cache_length]

  # Context portion: same window for ALL canvas tokens.
  # Attend to context positions in [context_start, context_end).
  context_mask = (cache_indices >= context_start[:, None]) & (
      cache_indices < context_end[:, None]
  )

  # Canvas portion: all canvas tokens attend to all other canvas tokens.
  # Canvas is written at [num_valid_tokens, num_valid_tokens + canvas_length).
  canvas_end = jnp.minimum(physical_valid_tokens + canvas_length, cache_length)
  canvas_mask = (cache_indices >= physical_valid_tokens[:, None]) & (
      cache_indices < canvas_end[:, None]
  )

  # Combine: attend to context window OR canvas self-attention.
  combined = context_mask | canvas_mask  # [B, cache_length]

  return jnp.broadcast_to(
      combined[:, None, :], (batch_size, canvas_length, cache_length)
  )
