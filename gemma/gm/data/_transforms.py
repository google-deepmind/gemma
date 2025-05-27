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

"""Data processing ops."""

from __future__ import annotations

import dataclasses
import textwrap

from gemma.gm.data import _functional
from gemma.gm.text import _tokenizer
from grain import python as grain
from kauldron import kd
from kauldron.typing import Array  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(kw_only=True, frozen=True)
class DecodeBytes(kd.data.ElementWiseTransform):
  """Decode `bytes` to `str`."""

  encoding: str = "utf-8"

  def map_element(self, element):
    return element.decode(self.encoding)


@dataclasses.dataclass(kw_only=True, frozen=True)
class FormatText(kd.data.ElementWiseTransform):
  """Equivalent to `template.format(text=my_string)`.

  Attributes:
    template: The template containing the `{text}` placeholder. Note that the
      template is detented (but not stripped).
  """

  template: str

  def __post_init__(self):
    super().__post_init__()
    if "{text}" not in self.template:
      raise ValueError(
          f"Template must contain '{{text}}' placeholder, got {self.template!r}"
      )
    object.__setattr__(self, "template", textwrap.dedent(self.template))

  def map_element(self, element):
    return self.template.format(text=element)


@dataclasses.dataclass(kw_only=True, frozen=True)
class Tokenize(kd.data.ElementWiseTransform):
  """Tokenize a string to ids.

  Attributes:
    tokenizer: The tokenizer to use.
    add_eos: Whether to add the EOS token (`1`) to the end of the sequence.
    add_bos: Whether to add the BOS token (`2`) to the beginning of the
      sequence.
  """

  tokenizer: _tokenizer.Tokenizer
  add_eos: bool = False
  add_bos: bool = False

  def map_element(self, element: str):
    return self.tokenizer.encode(
        element,
        add_bos=self.add_bos,
        add_eos=self.add_eos,
    )


@dataclasses.dataclass(kw_only=True, frozen=True)
class Pad(kd.data.ElementWiseTransform):
  """Add zeros to the end of the sequence to reach the max length.

  Attributes:
    max_length: The max length of the sequence.
    truncate: Whether to truncate the sequence to the max length. If `False`,
      sequences longer than the `max_length` will raise an error.
  """

  max_length: int
  # TODO(epot): Should this be another transform instead ?
  truncate: bool = False

  # Do not @typechecked as `element` can be `list` too.
  def map_element(self, element: Array["length"]) -> Array["max_length"]:
    return _functional.pad(
        element,
        max_length=self.max_length,
        truncate=self.truncate,
    )


@dataclasses.dataclass(kw_only=True, frozen=True)
class MapInts(kd.data.ElementWiseTransform):
  """Replace each int by a new value."""

  old_to_new: dict[int, int]

  def map_element(self, element):
    try:
      return self.old_to_new[element]
    except KeyError:
      raise KeyError(f"Label {element} not found in `old_to_new`.") from None


@dataclasses.dataclass(kw_only=True, frozen=True)
class AddSeq2SeqFields(grain.MapTransform):
  """Adds the model `input`, `target` and `loss_mask`.

  From prompt and response token ids, generate the model `input`, `target` and
  `loss_mask`.

  Example:

  ```python
  # Input:
  {
      'prompt': [10, 11, 12, 13],
      'response': [20, 21, 1],  # Here, response ends with EOS token.
  }
  # Ouptut:
  {
      'input':       [10, 11, 12, 13, 20, 21],
      'target':      [11, 12, 13, 20, 21,  1],
      'target_mask': [ 0,  0,  0,  1,  1,  1],
  }
  ```

  Note:
    - Input and target are the same sequence shifted by one token.
    - The last token from the target is truncated from the input (as there's no
      target for it)

  Attributes:
    in_prompt: Input key
    in_response: Input key
    out_input: Output key (will be added to the example dict)
    out_target: Output key (will be added to the example dict)
    out_target_mask: Output key (will be added to the example dict)
  """

  in_prompt: kd.kontext.Key  # e.g. `'prompt'`
  in_response: kd.kontext.Key  # e.g. `'response'`

  out_input: kd.kontext.Key  # e.g. `'input'`
  out_target: kd.kontext.Key  # e.g. `'target'`
  out_target_mask: kd.kontext.Key  # e.g. `'target_mask'`

  # Should we allow to customize the last-token to allow `MASKED`, `EOS` or
  # `TRUNCATED` ?

  def map(self, element):
    # Extract the values from the `dict` example.
    # `kontext.get_by_path(element, self.in_prompt)` is equivalent to
    # `element[self.in_prompt]`, but supports nested dicts and dataclasses.
    prompt_tokens = kd.kontext.get_by_path(element, self.in_prompt)
    response_tokens = kd.kontext.get_by_path(element, self.in_response)

    out = _functional.make_seq2seq_fields(
        prompt=prompt_tokens,
        response=response_tokens,
    )

    # Add the fields to the output `dict`.
    # Equivalent to `element[self.out_input] = ...`
    kd.kontext.set_by_path(element, self.out_input, out.input)
    kd.kontext.set_by_path(element, self.out_target, out.target)
    kd.kontext.set_by_path(element, self.out_target_mask, out.target_mask)
    return element
