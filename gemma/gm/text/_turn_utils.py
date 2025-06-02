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

"""Last turn utils."""

import flax
from gemma.gm.data import _functional
from gemma.gm.nn import _config
from gemma.gm.text import _sampler_loop
import jax.numpy as jnp
from kauldron.typing import Bool, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@flax.struct.dataclass(kw_only=True)
class PrevTurns:
  """Wrapper around the last state to help compute multi-turn constants."""

  last_state: _sampler_loop.SamplingState | None

  @property
  def cache(self) -> _config.Cache:
    assert self.last_state is not None
    return self.last_state.cache

  @property
  def last_token_pos(self) -> Int['#B']:
    """Offset of the last predicated token position."""
    if self.last_state is None:
      return jnp.zeros((1,), dtype=jnp.int32)
    else:
      return self.last_state.last_token_pos

  @property
  def used_cache_length(self) -> int:
    if self.last_state is None:
      return 0
    else:
      return int(self.last_state.used_cache_length) + 1

  @typechecked
  def make_prefill_attention_mask(
      self,
      *,
      next_turn_attention_mask: Bool['B L L'],
      prefill_cache_length: int,
      # L_with_prev_turns is: {self.used_cache_length}+L+padding
  ) -> Bool['B L L_with_prev_turns']:
    """Make the attention mask for the next turn."""
    if self.last_state is None:
      return next_turn_attention_mask

    # Eventually, add the previous turns attention mask.
    # Make the attention mask for the KV cache.

    _, next_prompt_length, next_prompt_length2 = next_turn_attention_mask.shape
    assert next_prompt_length == next_prompt_length2

    # b 1 used_cache_length
    prev_attention_mask = self.prev_attention_mask[:, None, :]
    prev_attention_mask = jnp.broadcast_to(
        prev_attention_mask,
        (
            prev_attention_mask.shape[0],  # b
            next_prompt_length,  # L
            prev_attention_mask.shape[2],  # used_cache_length
        ),
    )

    attention_mask = jnp.concat(
        [prev_attention_mask, next_turn_attention_mask], axis=-1
    )

    attention_mask = _functional.pad(
        attention_mask,
        max_length=prefill_cache_length,
    )

    return attention_mask

  @property
  def prev_attention_mask(self) -> Bool['B {self.used_cache_length}']:
    assert self.last_state is not None
    # The full_attention_mask from the last turn is padded with zeros
    # as post-processing step in the sampler loop.
    # (i.e. if end_tokens is predicted, all tokens after are masked).
    return self.last_state.full_attention_mask[:, : self.used_cache_length]

  def __bool__(self) -> bool:
    return self.last_state is not None
