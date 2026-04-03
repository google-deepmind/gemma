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

"""Gemma4 sampler with variable-aspect-ratio image and audio support."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools

from gemma.gm.data import _functional
from gemma.gm.nn import _transformer_like
from gemma.gm.nn.gemma4 import _transformer as gemma4_transformer
from gemma.gm.nn.gemma4.vision import _preprocessing
from gemma.gm.text import _prefill
from gemma.gm.text import _sampler
from gemma.gm.text import _sampler_loop
from gemma.gm.text import _sampling
from gemma.gm.text import _tokenizer
from gemma.gm.typing import _common
from gemma.gm.utils import _types
from gemma.gm.vision import _token_utils
import jax
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import PRNGKeyLike
import numpy as np
from PIL import Image


@dataclasses.dataclass(frozen=True, kw_only=True)
class Gemma4Sampler:
  """Sampler for Gemma4 with variable-aspect-ratio image support.

  This sampler handles preprocessing of variable-size images internally,
  expanding each `<|image|>` token to the correct number of placeholder
  tokens based on each image's actual dimensions.

  Attributes:
    model: Gemma transformer model.
    params: Model parameters.
    tokenizer: Tokenizer.
    sampling: Sampling method. Default to greedy.
    forbidden_tokens: List of tokens forbidden from generation.
    stop_tokens: List of tokens that stop generation.
    cache_length: Maximum cache length.
    max_out_length: Maximum output buffer length.
    pad_length: Pad lengths for static shapes.
    patch_size: Patch size for vision encoder.
    max_soft_tokens: Maximum soft tokens per image.
    pooling_kernel_size: Pooling kernel size.
    audio_sample_rate: Audio sample rate in Hz.
    audio_seq_length: Maximum audio sequence length.
  """

  model: _transformer_like.TransformerLike
  params: _common.Params
  tokenizer: _tokenizer.Tokenizer = None  # pytype: disable=annotation-type-mismatch
  sampling: _sampling.SamplingMethod = dataclasses.field(
      default_factory=_sampling.Greedy
  )
  forbidden_tokens: Sequence[str | int] | None = None
  stop_tokens: Sequence[str | int] | None = None
  cache_length: int = 4096
  max_out_length: int = 2048
  pad_length: None | int | tuple[int, ...] = (256, 512, 1024)
  patch_size: int = 16
  max_soft_tokens: int = 1120
  pooling_kernel_size: int = 3
  audio_sample_rate: int = 16000
  audio_seq_length: int = 750

  def __post_init__(self):
    if self.tokenizer is None:
      if not self.model.INFO.tokenizer_version:
        raise ValueError(
            "The model does not specify a tokenizer to use. "
            "Please explicitly set the tokenizer argument."
        )
      object.__setattr__(
          self,
          "tokenizer",
          _tokenizer.Tokenizer.from_version(self.model.INFO.tokenizer_version),
      )

  def sample(
      self,
      prompt: str | Sequence[str],
      *,
      images: list[np.ndarray | Image.Image] | None = None,
      audio: list[np.ndarray] | None = None,
      audio_lengths: list[int] | None = None,
      max_new_tokens: int | None = None,
      sampling: _sampling.SamplingMethod | None = None,
      rng: PRNGKeyLike | None = None,
      return_state: bool = False,
      last_state: _sampler_loop.SamplingState | None = None,
      sharding: kd.sharding.ShardingTree | None = None,
  ) -> str | list[str] | _sampler.SamplerOutput:
    """Samples from the model with variable-aspect-ratio image support.

    Args:
      prompt: Text prompt(s). Use `<|image|>` as image placeholder.
      images: List of raw images (numpy arrays or PIL Images) of any size.
      audio: List of audio arrays or None.
      audio_lengths: List of audio lengths or None.
      max_new_tokens: Maximum new tokens to generate.
      sampling: Sampling method override.
      rng: Random seed or PRNGKey.
      return_state: If True, return SamplerOutput with state.
      last_state: Previous state for multi-turn.
      sharding: Sharding tree.

    Returns:
      The sampled output.
    """
    sampling = sampling or self.sampling
    rng = _sampler._normalize_rng(rng)  # pylint: disable=protected-access

    has_batch_dim = _sampler._get_has_batch_dim(prompt)  # pylint: disable=protected-access

    prompt_list = _sampler._normalize_prompt(  # pylint: disable=protected-access
        prompt, format=self.tokenizer.FORMAT
    )
    tokens_list = [
        self.tokenizer.encode(p, add_bos=last_state is None)
        for p in prompt_list
    ]

    vision_input = None
    soft_token_counts = []
    if images:
      vision_input = self._preprocess_images(images)
      soft_token_counts = list(vision_input.soft_token_counts)

    max_prompt_len = max(len(t) for t in tokens_list)
    tokens = _functional.pad(tokens_list, max_length=max_prompt_len)
    tokens = np.asarray(tokens)

    if soft_token_counts:
      tokens = _token_utils.add_variable_extra_tokens_for_images(
          tokens,
          soft_token_counts=soft_token_counts,
      )

    audio_soft_token_counts = []
    if audio:
      if audio_lengths is None:
        audio_lengths = [len(a) for a in audio]

      for length in audio_lengths:
        frame_length = int(round(self.audio_sample_rate * 20.0 / 1000.0))
        hop_length = int(round(self.audio_sample_rate * 10.0 / 1000.0))
        frame_size_for_unfold = frame_length + 1
        num_mel_frames = (length - frame_size_for_unfold) // hop_length + 1
        t = num_mel_frames
        for _ in range(2):
          t_padded = t + 2
          t = (t_padded - 3) // 2 + 1
        audio_soft_token_counts.append(min(t, self.audio_seq_length))

      tokens = _token_utils.add_variable_extra_tokens_for_audio(
          tokens,
          soft_token_counts=audio_soft_token_counts,
      )

      max_audio_len = max(len(a) for a in audio)
      padded_audio = np.zeros((len(audio), max_audio_len), dtype=np.float32)
      for i, a in enumerate(audio):
        padded_audio[i, : len(a)] = a
      audio = jnp.asarray(padded_audio)[None]       # [1, N_audios, max_samples]
      audio_lengths = jnp.asarray(audio_lengths)[None]  # [1, N_audios]

    tokens = jnp.asarray(tokens)
    if sharding is not None:
      tokens = kd.sharding.device_put(tokens, sharding)

    input_config = self.model.config.input_config

    if images:
      input_config = _types.InputConfig(
          support_images=True,
          num_tokens_per_image=input_config.num_tokens_per_image,
          special_tokens=input_config.special_tokens,
          already_expanded=True,
      )

    inputs = _types.Input(
        text=tokens,
        images=None,
        config=input_config,
    )

    init_state = _prefill.prefill(
        model=self.model,
        params=self.params,
        input=inputs,
        last_state=last_state,
        cache_length=self.cache_length,
        pad_length=self.pad_length,
        rng=rng,
        sharding=sharding,
        max_out_length=self.max_out_length,
        vision_input=vision_input,
        audio=audio,
        audio_lengths=audio_lengths,
        audio_soft_token_counts=tuple(audio_soft_token_counts),
    )

    if max_new_tokens and max_new_tokens > self.max_out_length:
      raise ValueError(
          "max_new_tokens should be smaller or equal to max_out_length. Got:"
          f" {max_new_tokens} / {self.max_out_length}"
      )
    max_new_tokens = max_new_tokens or self.max_out_length
    max_new_tokens = jnp.asarray(max_new_tokens)

    sampler = _sampler_loop.SamplerLoop(
        model=self.model,
        end_tokens=(
            self.tokenizer.special_tokens.EOS,
            self.tokenizer.special_tokens.END_OF_TURN,
            self.tokenizer.special_tokens.BEGIN_OF_TOOL_RESPONSE,
            *self._normalized_stop_tokens,
        ),
        forbidden_tokens=self._normalized_forbidden_tokens,
        sampling=sampling,
        cache_length=self.cache_length,
        special_tokens=self.tokenizer.special_tokens,
    )

    state = sampler.sample(
        params=self.params,
        init_state=init_state,
        max_new_tokens=max_new_tokens,
        stream=False,
    )

    predicted_tokens = state.predicted_tokens
    if jax.process_count() > 1:
      predicted_tokens = kd.sharding.with_sharding_constraint(
          predicted_tokens,
          kd.sharding.REPLICATED,
      )

    predicted_texts = [self.tokenizer.decode(t) for t in predicted_tokens]

    if not has_batch_dim:
      (predicted_texts,) = predicted_texts

    if return_state:
      return _sampler.SamplerOutput(
          text=predicted_texts,
          state=state,
      )
    else:
      return predicted_texts

  def _preprocess_images(
      self,
      images: list[np.ndarray | Image.Image],
  ) -> gemma4_transformer.PreprocessedVisionInput:
    """Preprocess images."""
    patches, positions_xy, soft_token_counts = (
        _preprocessing.preprocess_and_patchify(
            images,
            patch_size=self.patch_size,
            max_soft_tokens=self.max_soft_tokens,
            pooling_kernel_size=self.pooling_kernel_size,
        )
    )
    n_images, max_patches = patches.shape[0], patches.shape[1]
    patches = jnp.reshape(
        patches, (1, n_images * max_patches, patches.shape[2])
    )
    positions_xy = jnp.reshape(
        positions_xy, (1, n_images * max_patches, positions_xy.shape[2])
    )
    return gemma4_transformer.PreprocessedVisionInput(
        patches=patches,
        positions_xy=positions_xy,
        soft_token_counts=tuple(soft_token_counts),
    )

  @functools.cached_property
  def _normalized_forbidden_tokens(self) -> tuple[int, ...] | None:
    forbidden_tokens = self._normalize_tokens(self.forbidden_tokens)
    forbidden_tokens += self.tokenizer.FORBIDDEN_TOKENS
    return forbidden_tokens

  @functools.cached_property
  def _normalized_stop_tokens(self) -> tuple[int, ...]:
    return self._normalize_tokens(self.stop_tokens)

  def _normalize_tokens(
      self, tokens: Sequence[str | int] | None
  ) -> tuple[int, ...]:
    if tokens is None:
      return ()
    else:
      return tuple(_sampler._normalize_token(self.tokenizer, t) for t in tokens)  # pylint: disable=protected-access
