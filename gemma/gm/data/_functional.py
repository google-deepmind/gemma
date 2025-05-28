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

"""Functional version of the `gm.data` transforms."""

from etils import enp
import flax
import jax
from kauldron.typing import Array, Bool, Int, PyTree  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np


# Do not @typechecked as `element` can be `list` too.
def pad(
    element: PyTree[Array["sequence"]],
    max_length: int,
    *,
    truncate: bool = False,
    fill_value: int = 0,
    axis: int = -1,
) -> PyTree[Array["max_length"]]:
  """Add zeros to the end of the sequence to reach the max length.

  Supports padding multiple arrays at once.

  Args:
    element: The sequence to pad.
    max_length: The max length of the sequence.
    truncate: Whether to truncate the sequence to the max length. If `False`,
      sequences longer than the `max_length` will raise an error.
    fill_value: The value to fill the padded sequence with.
    axis: The axis in which to pad the sequence (only -1 supported at the
      moment).

  Returns:
    The padded sequence of length `max_length`.
  """
  if axis != -1:
    raise NotImplementedError("Only `axis=-1` is supported.")
  return jax.tree.map(
      lambda x: _pad(
          x,
          max_length=max_length,
          fill_value=fill_value,
          truncate=truncate,
      ),
      element,
      is_leaf=_is_list_array,  # Also supports `[0, 1, ...]`
  )


def _pad(
    element: Array["sequence"],
    *,
    max_length: int,
    fill_value: int,
    truncate: bool = False,
) -> Array["max_length"]:
  """Inner padding implementation."""
  # Use `xnp` so it supports both `np` and `jnp`.
  xnp = enp.lazy.get_xnp(element, strict=False)

  element = xnp.asarray(element)

  # TODO(epot): Could add an `axis=` kwarg to support multi-dimensional arrays.
  seq_length = element.shape[-1]
  if not truncate and seq_length > max_length:
    raise ValueError(
        f"Cannot pad sequence of length {seq_length}. Is longer than the"
        f" max length {max_length}. Set `truncate=True`."
    )
  sentence_tokens = element[..., :max_length]

  pad_width = [(0, 0)] * (sentence_tokens.ndim - 1) + [
      (0, max_length - sentence_tokens.shape[-1])
  ]
  return xnp.pad(sentence_tokens, pad_width, constant_values=fill_value)


@flax.struct.dataclass
class Seq2SeqFields:
  """Fields for next token prediction."""

  input: Int["*b l"]
  target: Int["*b l"]
  target_mask: Bool["*b l"]


# Note: There's no `batch` dimension here. It wouldn't make much sense as
# each example has a different length, so batching can only be applied
# after the output is padded.
def make_seq2seq_fields(
    prompt: Int["prompt_len"],
    response: Int["response_len"],
) -> Seq2SeqFields:
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
  sequence = np.concatenate([prompt, response])

  # Create the loss mask.
  target_mask = np.concatenate([
      np.zeros((len(prompt) - 1,), dtype=np.bool_),
      np.ones((len(response),), dtype=np.bool_),
  ])

  return Seq2SeqFields(
      input=sequence[:-1],
      target=sequence[1:],
      target_mask=target_mask,
  )


def _is_list_array(x) -> bool:
  """Returns `True` if `x` is a list of ints, like `[0, 1, ...]`."""
  return isinstance(x, list | tuple) and all(isinstance(x, int) for x in x)
