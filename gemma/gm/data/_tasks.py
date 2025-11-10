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

"""End-to-end transformations."""

from __future__ import annotations

import dataclasses

import einops
from etils.etree import jax as etree  # pylint: disable=g-importing-member
from gemma.gm.data import _functional
from gemma.gm.text import _template
from gemma.gm.text import _tokenizer
from grain import python as grain
import jax
from kauldron import kd
import numpy as np


@dataclasses.dataclass(kw_only=True, frozen=True)
class Seq2SeqTask(grain.MapTransform):
  """Sequence-to-sequence task.

  This task will:

  * Format the prompt and response to match the expected dialog template (i.e.
    add the `<start_of_turn>user`, `<end_of_turn>`,...)
  * Tokenize the prompt and response.
  * Concatenate the input and response to create the model input and target
    (target is the input shifted by one token).
  * Create the loss mask (0 for prompt, 1 for response)
  * Pad/truncate the input and target to the max length.

  Example:

  ```python
  # Input:
  {
      'prompt': 'Hello! I would love to visit France.',
      'response': 'Bonjour ! J'adorerais visiter la France.',
  }
  # Ouptut:
  {
      'input': i32['max_length'],
      'target': i32['max_length 1'],
      'target_mask': bool['max_length 1'],
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
    drop_inputs: If True, remove the input keys from the output.
    max_length: The max length of the sequence (examples will be
      padded/truncated to this length).
    truncate: Whether to truncate the sequence to the max length. If `False`,
      sequences longer than the `max_length` will raise an error.
    sampling: If `True`, the dataset will yield the original prompt and response
      so they can be used inside `gm.evals.SamplerEvaluator`.
  """

  in_prompt: kd.kontext.Key  # e.g. `'prompt'`
  in_response: kd.kontext.Key  # e.g. `'response'`

  out_input: kd.kontext.Key  # e.g. `'input'`
  out_target: kd.kontext.Key  # e.g. `'target'`
  out_target_mask: kd.kontext.Key  # e.g. `'target_mask'`

  drop_inputs: bool = True

  tokenizer: _tokenizer.Tokenizer

  # Padding parameters
  max_length: int
  truncate: bool = False

  sampling: bool = False

  def map(self, element):
    # Deep-copy to avoid mutating the input.
    element = etree.copy(element)

    # Extract the values from the `dict` example.
    # `kontext.get_by_path(element, self.in_prompt)` is equivalent to
    # `element[self.in_prompt]`, but supports nested dicts and dataclasses.
    prompt = kd.kontext.get_by_path(element, self.in_prompt)
    response = kd.kontext.get_by_path(element, self.in_response)

    # TODO(epot): Supports nested drop
    if self.drop_inputs:
      del element[self.in_prompt]
      del element[self.in_response]

    # Some datasets (TFDS) returns `bytes` instead of `str`, so decode them.
    prompt = _decode_bytes(prompt)
    response = _decode_bytes(response)

    # Format the input to match the expected dialog template.
    # TODO(epot): Add a `template` protocol to allow customizing this.
    prompt = _template.PROMPT.format(prompt)
    response = _template.ANSWER.format(response)

    # For sampling, we don't need to tokenize the input.
    if self.sampling:
      kd.kontext.set_by_path(element, self.out_input, prompt)
      kd.kontext.set_by_path(element, self.out_target, response)
      return element

    # Tokenize the input and the responses.
    prompt = self.tokenizer.encode(prompt, add_bos=True)
    response = self.tokenizer.encode(response)

    # Create the model inputs/targets/loss_mask.
    out = _functional.make_seq2seq_fields(
        prompt=prompt,
        response=response,
    )

    # Add padding.
    out = _functional.pad(
        out,
        max_length=self.max_length,
        truncate=self.truncate,
    )

    # For shape compatibility with the loss
    target = einops.rearrange(out.target, "... -> ... 1")
    target_mask = einops.rearrange(out.target_mask, "... -> ... 1")

    # Add the fields to the output `dict`.
    # Equivalent to `element[self.out_input] = ...`
    kd.kontext.set_by_path(element, self.out_input, out.input)
    kd.kontext.set_by_path(element, self.out_target, target)
    kd.kontext.set_by_path(element, self.out_target_mask, target_mask)
    return element


@dataclasses.dataclass(kw_only=True, frozen=True)
class ContrastiveTask(grain.MapTransform):
  """Creates the contrastive model inputs for DPO-like loss.

  Input:

  ```python
  {
      'prompt': 'How much are 2+2 ?',
      'chosen': 'Yes, this is 4.',
      'rejected': 'Of course, 2+2 is 42.',
  }
  ```

  Output:

  ```python
  {
      'tokens': i32['2 max_length'],
      'mask': bool['2 max_length'],
  }
  ```

  In the output, `[chosen, rejected]` token ids are stacked (in that order).

  Attributes:
    in_prompt: Input key
    in_chosen: Input key
    in_rejected: Input key
    out_tokens: Output key (will be added to the example dict)
    out_mask: Output key (will be added to the example dict)
    tokenizer: The tokenizer to use.
    max_length: The max length of the sequence (examples will be
      padded/truncated to this length).
    truncate: Whether to truncate the sequence to the max length. If `False`,
      sequences longer than the `max_length` will raise an error.
    drop_inputs: If True, remove the input keys from the output.
  """

  in_prompt: kd.kontext.Key  # e.g. `'input'`
  in_chosen: kd.kontext.Key  # e.g. `'chosen'`
  in_rejected: kd.kontext.Key  # e.g. `'rejected'`

  out_tokens: kd.kontext.Key  # e.g. `'tokens'`
  out_targets: kd.kontext.Key  # e.g. `'target'`
  out_mask: kd.kontext.Key  # e.g. `'mask'`

  tokenizer: _tokenizer.Tokenizer

  max_length: int
  truncate: bool = False

  drop_inputs: bool = True

  def map(self, element):
    prompt = kd.kontext.get_by_path(element, self.in_prompt)
    chosen = kd.kontext.get_by_path(element, self.in_chosen)
    rejected = kd.kontext.get_by_path(element, self.in_rejected)

    # Some datasets (TFDS) returns `bytes` instead of `str`, so decode them.
    prompt = _decode_bytes(prompt)
    chosen = _decode_bytes(chosen)
    rejected = _decode_bytes(rejected)

    # Format the input to match the expected dialog template.
    # TODO(epot): Move this in a separate FormatDialog transform.
    prompt = _template.PROMPT.format(prompt)
    chosen = _template.ANSWER.format(chosen)
    rejected = _template.ANSWER.format(rejected)

    # Tokenize the input and the responses.
    # Note: Input should start with begin-of-sequence token.
    prompt = self.tokenizer.encode(prompt, add_bos=True)
    chosen = self.tokenizer.encode(chosen)
    rejected = self.tokenizer.encode(rejected)

    next_token_chosen = _functional.make_seq2seq_fields(
        prompt=prompt,
        response=chosen,
    )
    next_token_rejected = _functional.make_seq2seq_fields(
        prompt=prompt,
        response=rejected,
    )

    # Add padding.
    (next_token_chosen, next_token_rejected) = _functional.pad(
        (next_token_chosen, next_token_rejected),
        max_length=self.max_length,
        truncate=self.truncate,
    )

    # Stack the input and target.
    out = jax.tree.map(
        lambda x, y: np.stack([x, y], axis=0),
        next_token_chosen,
        next_token_rejected,
    )

    # Add the fields to the output `dict`.
    # Equivalent to `element[self.out_input] = ...`
    kd.kontext.set_by_path(element, self.out_tokens, out.input)
    kd.kontext.set_by_path(element, self.out_targets, out.target)
    kd.kontext.set_by_path(element, self.out_mask, out.target_mask)

    # TODO(epot): Supports nested drop
    if self.drop_inputs:
      del element[self.in_prompt]
      del element[self.in_chosen]
      del element[self.in_rejected]

    return element


def _decode_bytes(element):
  if isinstance(element, bytes):
    return element.decode("utf-8")
  else:
    return element
