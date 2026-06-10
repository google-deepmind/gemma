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

"""Data processing ops."""

from __future__ import annotations

import dataclasses

from grain import python as grain
import numpy as np


@dataclasses.dataclass(kw_only=True, frozen=True)
class CopyField(grain.MapTransform):
  """Copy a field to a new key. Useful for stashing a value before mutation."""

  src_key: str
  dst_key: str

  def map(self, features):
    features[self.dst_key] = features[self.src_key]
    return features


@dataclasses.dataclass(kw_only=True, frozen=True)
class CanvasChunker(grain.MapTransform):
  """Chunk response tokens into fixed-size canvases (flattened output).

  Reads the tokenised response and produces:
    - canvas: int array of shape (num_canvases * canvas_size,)
        Flat sequence of canvas tokens. Valid canvases contain response tokens
        (with EOS fill at the end of partial canvases). Invalid canvases are
        filled with PAD tokens.
    - canvas_id: int array of shape (num_canvases * canvas_size,)
        Per-token canvas index (0-based).
        Example for 3 canvases of size 4.
          [0, 0, 0, 0,  1, 1, 1, 1,  2, 2, 2, 2]
    - canvas_mask: bool array of shape (num_canvases * canvas_size,)
        True for tokens in valid canvases, False for pure-PAD canvas tokens.
  """

  in_response: str = "response"
  out_canvas: str = "canvas"
  out_canvas_id: str = "canvas_id"
  out_canvas_mask: str = "canvas_mask"

  num_canvases: int = 4
  canvas_size: int = 64
  eos_token: int = 1
  pad_token: int = 0

  def map(self, features):
    response = np.asarray(features[self.in_response])  # 1-D, variable length
    total_capacity = self.num_canvases * self.canvas_size

    # --- Truncate if response is too long ---
    response_truncated = response[:total_capacity]
    num_response_tokens = len(response_truncated)

    # --- Build the flat canvas array ---
    # Start with a PAD-filled array.
    flat = np.full(total_capacity, self.pad_token, dtype=response.dtype)
    flat[:num_response_tokens] = response_truncated

    # --- EOS-fill the partially filled final canvas ---
    # Only the *last canvas that has real tokens* needs EOS fill;
    # canvases after it stay as PAD.
    if num_response_tokens > 0:
      last_canvas_idx = (num_response_tokens - 1) // self.canvas_size
      canvas_end = (last_canvas_idx + 1) * self.canvas_size
      flat[num_response_tokens:canvas_end] = self.eos_token
      num_valid_canvases = last_canvas_idx + 1
    else:
      num_valid_canvases = 0

    # --- Build canvas_id: per-token canvas index ---
    # Example (3 canvases, size 4):
    #   [0, 0, 0, 0,  1, 1, 1, 1,  2, 2, 2, 2]
    canvas_id = np.repeat(np.arange(self.num_canvases), self.canvas_size)

    # --- Build per-token canvas mask ---
    # True for valid-canvas tokens, False for pure-PAD canvas tokens.
    # Note the last canvas with EOS tokens is still valid.
    canvas_mask = np.zeros(total_capacity, dtype=np.bool_)
    canvas_mask[: num_valid_canvases * self.canvas_size] = True

    features[self.out_canvas] = flat
    features[self.out_canvas_id] = canvas_id
    features[self.out_canvas_mask] = canvas_mask
    return features


@dataclasses.dataclass(kw_only=True, frozen=True)
class SequenceTargetShift(grain.MapTransform):
  """Shift the full sequence to create targets for the encoder loss.

  Reads prompt and canvas tokens, concatenates them, and produces:
    - encoder_target: Shifted forward by 1.
    - encoder_target_mask: True for positions where target is a valid token.
  """

  in_prompt: str = "prompt"
  in_canvas: str = "canvas"
  in_canvas_mask: str = "canvas_mask"
  out_encoder_target: str = "encoder_target"
  out_encoder_target_mask: str = "encoder_target_mask"
  pad_token: int = 0

  def map(self, features):
    prompt = np.asarray(features[self.in_prompt])
    canvas = np.asarray(features[self.in_canvas])
    canvas_mask = np.asarray(features[self.in_canvas_mask])

    # Ensure they are 1D
    prompt = prompt.flatten()
    canvas = canvas.flatten()
    canvas_mask = canvas_mask.flatten()

    prompt_valid = prompt != self.pad_token
    canvas_valid = canvas_mask

    full_seq = np.concatenate([prompt, canvas])
    full_valid = np.concatenate([prompt_valid, canvas_valid])

    # Shift targets: targets[i] = full_seq[i+1]
    encoder_target = np.roll(full_seq, -1)
    encoder_target[-1] = self.pad_token

    # Shift valid mask to match targets.
    # Require BOTH position i (current) and position i+1 (target) to be valid.
    # This prevents computing loss at PAD positions (e.g., at the prompt-canvas
    # boundary where the next token is a valid canvas token but the current
    # hidden state is from a PAD token).
    shifted_valid = np.roll(full_valid, -1)
    shifted_valid[-1] = False
    encoder_target_mask = full_valid & shifted_valid

    features[self.out_encoder_target] = encoder_target
    features[self.out_encoder_target_mask] = encoder_target_mask
    return features


@dataclasses.dataclass(kw_only=True, frozen=True)
class ReformatPubMedQAAnswer(grain.MapTransform):
  """Append ``The answer is: {answer}`` to PubMedQA responses.

  For short-answer mode the raw response is just ``yes``/``no``/``maybe``.
  For long-answer mode it is a paragraph-length explanation.  In both cases
  we append the structured answer marker so the model learns to produce it,
  mirroring the ``"The answer is <N>"`` pattern used by GSM8K.

  After this transform:
    - short: ``"yes"`` → ``"The answer is: yes"``
    - long:  ``"DBE appears to be..."`` →
      ``"DBE appears to be... The answer is: yes"``
  """

  response_key: str = "response_text"
  answer_key: str = "short_answer"

  def map(self, features):
    answer = features[self.answer_key].strip().lower()
    response = features[self.response_key]
    if isinstance(response, bytes):
      response = response.decode("utf-8")
    features[self.response_key] = f"{response.strip()} The answer is: {answer}"
    return features
