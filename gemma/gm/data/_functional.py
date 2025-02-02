# Copyright 2024 DeepMind Technologies Limited.
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

"""Functional version of the `gm.data` transforms."""

from etils import enp
import flax
from kauldron.typing import Array, Bool, Int  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np


# Do not @typechecked as `element` can be `list` too.
def pad(
    element: Array["sequence"],
    *,
    max_length: int,
    truncate: bool = False,
) -> Array["max_length"]:
  """Add zeros to the end of the sequence to reach the max length.

  Args:
    element: The sequence to pad.
    max_length: The max length of the sequence.
    truncate: Whether to truncate the sequence to the max length. If `False`,
      sequences longer than the `max_length` will raise an error.

  Returns:
    The padded sequence of length `max_length`.
  """
  # Use `xnp` so it supports both `np` and `jnp`.
  xnp = enp.lazy.get_xnp(element, strict=False)

  # TODO(epot): Could add an `axis=` kwarg to support multi-dimensional arrays.
  seq_length = len(element)
  if not truncate and seq_length > max_length:
    raise ValueError(
        f"Cannot pad sequence of length {seq_length}. Is longer than the"
        f" max length {max_length}. Set `truncate=True`."
    )
  sentence_tokens = element[:max_length]
  return xnp.pad(sentence_tokens, (0, max_length - len(sentence_tokens)))


@flax.struct.dataclass
class NextTokenPredictionFields:
  """Fields for next token prediction."""

  input: Int["*b l"]
  target: Int["*b l"]
  target_mask: Bool["*b l"]


# Note: There's no `batch` dimension here. It wouldn't make much sense as
# each example has a different length, so batching can only be applied
# after the output is padded.
def make_next_token_prediction_fields(
    prompt: Int["prompt_len"],
    response: Int["response_len"],
) -> NextTokenPredictionFields:
  """Create the model `input`, `target` and `loss_mask`.

  From prompt and response token ids, generate the model `input`, `target` and
  `loss_mask`.

  Example:

  ```python
  # Input:
  prompt = [10, 11, 12, 13],
  response = [20, 21, 1],  # Here, response ends with EOS token.

  # Ouptut:
  out.input =       [10, 11, 12, 13, 20, 21],
  out.target =      [11, 12, 13, 20, 21,  1],
  out.target_mask = [ 0,  0,  0,  1,  1,  1],
  ```

  Note:
    - Input and target are the same sequence shifted by one token.
    - The last token from the target is truncated from the input (as there's no
      target for it)

  Args:
    prompt: The prompt tokens.
    response: The response tokens.

  Returns:
    The input, target and mask, all of length `prompt_len + response_len - 1`.
  """
  # Concatenate the prompt and response tokens.
  sequence = np.concat([prompt, response])

  # Create the loss mask.
  target_mask = np.concat([
      np.zeros((len(prompt) - 1,), dtype=np.bool),
      np.ones((len(response),), dtype=np.bool),
  ])

  return NextTokenPredictionFields(
      input=sequence[:-1],
      target=sequence[1:],
      target_mask=target_mask,
  )
