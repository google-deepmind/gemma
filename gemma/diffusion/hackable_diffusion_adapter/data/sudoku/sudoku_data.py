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

"""Generic Bagz data source and Sudoku parsing transform for Kauldron."""

import dataclasses
import functools
from typing import Any

from gemma import gm
from gemma.diffusion.hackable_diffusion_adapter.data import data as adapter_data
import grain.python as grain
import jax
from kauldron import kd
from kauldron import random
from kauldron.data.py import base as kd_base
import tensorflow as tf

import bagz


class PicklableBagzReader(grain.RandomAccessDataSource):
  """A picklable wrapper for bagz.Reader to support multiprocessing."""

  def __init__(self, bagz_path: str):
    self.bagz_path = bagz_path
    self._reader = None
    self._length = None

  @property
  def reader(self):
    if self._reader is None:
      self._reader = bagz.Reader(self.bagz_path)
    return self._reader

  def __len__(self):
    if self._length is None:
      self._length = len(self.reader)
    return self._length

  def __getitem__(self, index):
    return self.reader[index]

  def __getstate__(self):
    state = self.__dict__.copy()
    state['_reader'] = None
    return state


@dataclasses.dataclass(frozen=True)
class Bagz(kd_base.DataSourceBase):
  """Generic Bagz dataset pipeline for Kauldron.

  Wraps `grain.BagDataSource` as a Kauldron `DataSourceBase`, allowing
  the use of standard Kauldron and Gemma data transforms.
  """

  bagz_path: str
  slice_start: int = 0
  slice_stop: int | None = None

  @functools.cached_property
  def data_source(self) -> grain.RandomAccessDataSource:
    return PicklableBagzReader(self.bagz_path)

  def ds_for_current_process(self, rng: random.PRNGKey) -> grain.MapDataset:
    ds = grain.MapDataset.source(self.data_source)
    ds = ds.seed(random.random_seed(rng))

    # Slice the dataset before sharding
    if self.slice_start != 0 or self.slice_stop is not None:
      ds = ds.slice(slice(self.slice_start, self.slice_stop))

    # Shard the dataset
    if self.shard_by_process:
      ds = ds[jax.process_index() :: jax.process_count()]

    # Global shuffle
    if self.shuffle:
      ds = ds.shuffle(
          seed=random.random_seed(random.fold_in_str(rng, "shuffle"))
      )
    return ds


@dataclasses.dataclass
class ParseSudokuExample(grain.MapTransform):
  """Parses raw Bagz bytes (tf.train.Example) into puzzle and solution."""

  def map(self, record_bytes: bytes) -> dict[str, Any]:
    example = tf.train.Example()
    example.ParseFromString(record_bytes)
    features = example.features.feature

    puzzle = features["puzzle"].bytes_list.value[0].decode("utf-8")
    solution = features["solution"].bytes_list.value[0].decode("utf-8")

    return {"prompt": puzzle, "response": solution, "puzzle_raw": puzzle}


_DEFAULT_SUDOKU_PROMPT = (
    "<|turn>system Solve the following Sudoku puzzle. Empty cells are"
    " represented by 0. Output ONLY the solved puzzle immediately as"
    " a 9x9 grid of numbers separated by spaces. Do not include ####,"
    " explanations, or any other text.<turn|>\n<|turn>user"
    " {text}<turn|>\n<|turn>model\n"
)


def make_sudoku_ds(
    bagz_path: str,
    training: bool,
    batch_size: int,
    prompt_len: int,
    num_canvases: int,
    canvas_size: int,
    slice_start: int = 0,
    slice_stop: int | None = None,
    prompt_template: str = _DEFAULT_SUDOKU_PROMPT,
    num_workers: int = 16,
) -> Bagz:
  """Build the Sudoku dataset pipeline using standard Kauldron transforms.

  Args:
    bagz_path: Path to the Bagz dataset.
    training: Whether this is training or evaluation.
    batch_size: Per-device batch size.
    prompt_len: Maximum prompt token length.
    num_canvases: Number of canvas chunks for the response.
    canvas_size: Token length of each canvas chunk.
    slice_start: Start index for dataset slicing.
    slice_stop: Stop index for dataset slicing.
    prompt_template: Template for the sudoku prompt.  Must contain ``{text}``
      which will be replaced with the puzzle string.  Defaults to a system
      prompt instructing the model to solve the puzzle.
    num_workers: Number of workers for data loading.

  Returns:
    A ``Bagz`` dataset config.
  """
  tokenizer = gm.text.Gemma4Tokenizer()
  pad_token = gm.text.Gemma4Tokenizer.special_tokens.PAD
  eos_token = gm.text.Gemma4Tokenizer.special_tokens.EOS

  transforms = [
      # 1. Parse raw bytes
      ParseSudokuExample(),
  ]

  if not training:
    # (eval only) Copy raw response string BEFORE formatting CoT
    transforms.append(
        adapter_data.CopyField(src_key="response", dst_key="solution_raw")
    )

  transforms.extend([
      # 2. Format Sudoku Response
      gm.data.FormatText(
          key="response",
          template="{text}",
      ),
      # 3. Format Prompt
      gm.data.FormatText(
          key="prompt",
          template=prompt_template,
      ),
      # 4. Tokenize
      gm.data.Tokenize(tokenizer=tokenizer, key="prompt", add_bos=True),
      gm.data.Tokenize(tokenizer=tokenizer, key="response"),
  ])

  if not training:
    transforms.extend([
        # (eval only) Tokenize raw solution
        gm.data.Tokenize(tokenizer=tokenizer, key="solution_raw"),
    ])

  transforms.extend([
      # 5. Pad prompt
      gm.data.Pad(key="prompt", max_length=prompt_len, truncate=False),
  ])

  if not training:
    # (eval only) Pad raw response tokens for accuracy metric.
    # Also preserve the raw puzzle tokens so we can identify masked cells.
    transforms.extend([
        gm.data.Pad(key="solution_raw", max_length=256, truncate=True),
        kd.data.Elements(rename={"solution_raw": "solution_tokens"}),
        gm.data.Tokenize(tokenizer=tokenizer, key="puzzle_raw"),
        gm.data.Pad(key="puzzle_raw", max_length=256, truncate=True),
        kd.data.Elements(rename={"puzzle_raw": "puzzle_tokens"}),
    ])

  transforms.extend([
      # 6. Canvas Chunker
      adapter_data.CanvasChunker(
          in_response="response",
          out_canvas="canvas",
          out_canvas_id="canvas_id",
          out_canvas_mask="canvas_mask",
          num_canvases=num_canvases,
          canvas_size=canvas_size,
          eos_token=eos_token,
          pad_token=pad_token,
      ),
      # 7. Sequence Target Shift
      adapter_data.SequenceTargetShift(
          pad_token=pad_token,
      ),
      # 8. Add trailing dimension to canvas (using standard Kauldron Rearrange)
      kd.data.Rearrange(key="canvas", pattern="c -> c 1"),
  ])

  keep_fields = [
      "prompt",
      "canvas",
      "canvas_id",
      "canvas_mask",
      "encoder_target",
      "encoder_target_mask",
  ]
  if not training:
    keep_fields.extend(["solution_tokens", "puzzle_tokens"])

  transforms.append(kd.data.Elements(keep=keep_fields))

  return Bagz(
      bagz_path=bagz_path,
      shuffle=training,
      num_epochs=None if training else 1,
      batch_size=batch_size,
      num_workers=num_workers,
      read_options=grain.ReadOptions(
          num_threads=16,
          prefetch_buffer_size=500,
      ),
      transforms=transforms,
      slice_start=slice_start,
      slice_stop=slice_stop,
  )
