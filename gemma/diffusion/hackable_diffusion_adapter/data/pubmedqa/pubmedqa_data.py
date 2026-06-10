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

"""PubMedQA data loader pipeline and custom PyGrain DataSource."""

from __future__ import annotations

import json

from etils import epath
from gemma import gm
from gemma.diffusion.hackable_diffusion_adapter.data import data as adapter_data
from kauldron import kd


class JsonlDataSource:
  """Standalone random-access data source backed by a JSONL file.

  Implements the ``grain.RandomAccessDataSource`` Protocol via structural
  typing (requires only ``__len__`` and ``__getitem__``).
  """

  def __init__(
      self, path: str, slice_start: int = 0, slice_stop: int | None = None
  ):
    self.path = path
    text = epath.Path(path).read_text()
    # Scientific abstracts often contain vertical tabs (\x0b) or form
    # feeds (\x0c), which splitlines() splits on, breaking JSON structures.
    # split('\n') is safe.
    self._data = [json.loads(line) for line in text.strip().split("\n")]
    if slice_start != 0 or slice_stop is not None:
      self._data = self._data[slice_start:slice_stop]

  def __len__(self) -> int:
    return len(self._data)

  def __getitem__(self, record_key: int):
    import jax  # pylint: disable=g-import-not-at-top

    ex = self._data[record_key]
    return jax.tree.map(lambda x: x, ex)  # deep-copy for mutation safety


def make_pubmedqa_ds(
    training: bool,
    batch_size: int,
    num_canvases: int,
    canvas_size: int,
    prompt_len: int,
    use_long_answer: bool = False,
    train_path: str = "",
    test_path: str = "",
    slice_start: int = 0,
    slice_stop: int | None = None,
    num_workers: int = 4,
):
  """Build the PubMedQA dataset pipeline.

  Args:
    training: Whether this is training or evaluation.
    batch_size: Per-device batch size.
    num_canvases: Number of canvas chunks for the response.
    canvas_size: Token length of each canvas chunk.
    prompt_len: Maximum prompt token length.
    use_long_answer: If True, use the ``long_response`` field (detailed
      explanation) as the training target.  Otherwise use the short ``response``
      field (yes/no/maybe).
    train_path: Path to the training JSONL file.
    test_path: Path to the test JSONL file.
    slice_start: Start index for dataset slicing.
    slice_stop: Stop index for dataset slicing.
    num_workers: Number of workers for data loading.

  Returns:
    A ``kd.data.py.DataSource`` dataset config.
  """
  tokenizer = gm.text.Gemma4Tokenizer()
  path = train_path if training else test_path
  response_field = "long_response" if use_long_answer else "response"

  transforms = [
      # Rename to canonical prompt / response keys.
      kd.data.Elements(rename={response_field: "response_text"}),
      # Append structured answer marker to response text so the model
      # learns to produce "The answer is: {yes|no|maybe}" at the end.
      adapter_data.ReformatPubMedQAAnswer(
          response_key="response_text",
          answer_key="short_answer",
      ),
      # Wrap in instruction-tuning turn tokens with a system prompt.
      gm.data.FormatText(
          key="prompt",
          template=(
              "<|turn>system\nYou are a medical research assistant. Given"
              " a medical research context and a question, provide a"
              " brief explanation and then state your final answer.\n\n"
              "When you show the final answer, use the expression"
              ' "The answer is: " followed by yes, no, or maybe.'
              " Use the expression without any modification.\n\n"
              "For example:\n"
              "The study shows a statistically significant improvement"
              " in outcomes for the treatment group compared to"
              " the control group (p < 0.05)."
              " The answer is: yes<turn|>\n"
              "<|turn>user\n{text}<turn|>\n<|turn>model\n"
          ),
      ),
      gm.data.FormatText(
          key="response_text",
          template="{text}<turn|>",
      ),
      # Tokenize prompt (with BOS) and response (without BOS).
      gm.data.Tokenize(tokenizer=tokenizer, key=["prompt"], add_bos=True),
      gm.data.Tokenize(tokenizer=tokenizer, key=["response_text"]),
      # Pad the prompt to a fixed maximum length.
      gm.data.Pad(key="prompt", max_length=prompt_len, truncate=True),
      # Chunk the response into multiple flattened canvases.
      adapter_data.CanvasChunker(
          in_response="response_text",
          out_canvas="canvas",
          out_canvas_id="canvas_id",
          out_canvas_mask="canvas_mask",
          num_canvases=num_canvases,
          canvas_size=canvas_size,
          eos_token=gm.text.Gemma4Tokenizer.special_tokens.EOS,
          pad_token=gm.text.Gemma4Tokenizer.special_tokens.PAD,
      ),
      # Shift the full sequence to create targets for AR loss.
      adapter_data.SequenceTargetShift(
          pad_token=gm.text.Gemma4Tokenizer.special_tokens.PAD,
      ),
      # Add a trailing dimension because the model expects it.
      kd.data.Rearrange(key="canvas", pattern="c -> c 1"),
  ]

  keep_fields = [
      "prompt",
      "canvas",
      "canvas_id",
      "canvas_mask",
      "encoder_target",
      "encoder_target_mask",
  ]

  if not training:
    # (eval only) Tokenize and pad short_answer for accuracy metrics.
    transforms.extend([
        gm.data.Tokenize(tokenizer=tokenizer, key=["short_answer"]),
        gm.data.Pad(key="short_answer", max_length=32, truncate=True),
        kd.data.Elements(rename={"short_answer": "short_answer_tokens"}),
    ])
    keep_fields.append("short_answer_tokens")

    if use_long_answer:
      # If long-answer config, copy and pad reference tokens for BLEU metric.
      transforms.extend([
          adapter_data.CopyField(
              src_key="response_text", dst_key="response_text_copy"
          ),
          gm.data.Pad(
              key="response_text_copy",
              max_length=num_canvases * canvas_size,
              truncate=True,
          ),
          kd.data.Elements(rename={"response_text_copy": "long_answer_tokens"}),
      ])
      keep_fields.append("long_answer_tokens")

  transforms.append(kd.data.Elements(keep=keep_fields))

  return kd.data.py.DataSource(
      data_source=JsonlDataSource(
          path=path, slice_start=slice_start, slice_stop=slice_stop
      ),
      shuffle=training,
      num_epochs=None if training else 1,
      batch_size=batch_size,
      num_workers=num_workers,
      transforms=transforms,
  )
