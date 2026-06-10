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

"""Convert Sudoku CSV dataset to Bagz format with space-separated digits."""

import csv
import os
from absl import app
from absl import flags
from absl import logging
import bagz
import tensorflow as tf

################################################################################
# MARK: Flags
################################################################################

_INPUT_CSV = flags.DEFINE_string(
    "input_csv",
    "",
    "Path to the input Sudoku CSV file.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "",
    "Directory to write the output Bagz files.",
)
_TRAIN_SPLIT = flags.DEFINE_float(
    "train_split",
    0.9,
    "Fraction of data to use for training.",
)
_MAX_RECORDS = flags.DEFINE_integer(
    "max_records",
    -1,
    "Maximum number of records to process. Use -1 for all.",
)


################################################################################
# MARK: Helpers
################################################################################


def make_tf_example(features_dict: dict[str, bytes]) -> tf.train.Example:
  """Create a tf.train.Example proto from a dictionary of byte features."""
  tf_features = {
      k: tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
      for k, v in features_dict.items()
  }
  return tf.train.Example(features=tf.train.Features(feature=tf_features))


def space_separate(s: str) -> str:
  """Insert space characters between every character in the string."""
  return " ".join(list(s))


def _write_records(records: list[tuple[str, str]], path: str) -> None:
  """Write the puzzle and solution records to a Bagz file."""
  with bagz.Writer(path) as writer:
    for puzzle, solution in records:
      puzzle_spaced = space_separate(puzzle)
      solution_spaced = space_separate(solution)
      features = {
          "puzzle": puzzle_spaced.encode("utf-8"),
          "solution": solution_spaced.encode("utf-8"),
      }
      example = make_tf_example(features)
      writer.write(example.SerializeToString())


################################################################################
# MARK: Main
################################################################################


def main(argv):
  """Parse a Sudoku CSV dataset and output train/eval Bagz files."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  input_path = _INPUT_CSV.value
  output_dir = _OUTPUT_DIR.value

  if not output_dir:
    raise ValueError("--output_dir must be specified.")

  os.makedirs(output_dir, exist_ok=True)

  train_path = os.path.join(output_dir, "sudoku_train.bagz")
  eval_path = os.path.join(output_dir, "sudoku_eval.bagz")

  logging.info("Reading from %s", input_path)
  records = []
  max_records = _MAX_RECORDS.value
  with open(input_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
      records.append((row["puzzle"], row["solution"]))
      if max_records > 0 and len(records) >= max_records:
        break

  num_records = len(records)
  logging.info("Found %d records", num_records)

  num_train = int(num_records * _TRAIN_SPLIT.value)
  train_records = records[:num_train]
  eval_records = records[num_train:]

  logging.info("Writing %d train records to %s", len(train_records), train_path)
  _write_records(train_records, train_path)

  logging.info("Writing %d eval records to %s", len(eval_records), eval_path)
  _write_records(eval_records, eval_path)

  logging.info("Conversion complete.")


if __name__ == "__main__":
  app.run(main)
