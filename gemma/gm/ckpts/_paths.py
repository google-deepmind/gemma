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

"""Checkpoint paths."""

import enum


class CheckpointPath(enum.StrEnum):
  """Hardcoded paths to Gemma checkpoints.

  Format: `{VERSION}_{SIZE}_{VARIANT}`.

  Variants:

  * `PT`: Pre-trained
  * `IT`: Instruction Tuned (most likely the model you want to use)

  For example, `GEMMA2_27B_IT` is Gemma V2, 27 Billion parameters, instruction
  tuned.
  """

  # ******** Gemma 2.0 ********
  # Pretrained
  GEMMA2_2B_PT = 'gs://gemma-data/checkpoints/gemma2-2b-pt'
  GEMMA2_9B_PT = 'gs://gemma-data/checkpoints/gemma2-9b-pt'
  GEMMA2_27B_PT = 'gs://gemma-data/checkpoints/gemma2-27b-pt'
  # Instruction Tuned
  GEMMA2_2B_IT = 'gs://gemma-data/checkpoints/gemma2-2b-it'
  GEMMA2_9B_IT = 'gs://gemma-data/checkpoints/gemma2-9b-it'
  GEMMA2_27B_IT = 'gs://gemma-data/checkpoints/gemma2-27b-it'

  # ******** Gemma 3.0 ********
  # Pretrained
  GEMMA3_270M_PT = 'gs://gemma-data/checkpoints/gemma3-270m-pt'
  GEMMA3_1B_PT = 'gs://gemma-data/checkpoints/gemma3-1b-pt'
  GEMMA3_4B_PT = 'gs://gemma-data/checkpoints/gemma3-4b-pt'
  GEMMA3_12B_PT = 'gs://gemma-data/checkpoints/gemma3-12b-pt'
  GEMMA3_27B_PT = 'gs://gemma-data/checkpoints/gemma3-27b-pt'
  # Instruction Tuned
  GEMMA3_270M_IT = 'gs://gemma-data/checkpoints/gemma3-270m-it'
  GEMMA3_1B_IT = 'gs://gemma-data/checkpoints/gemma3-1b-it'
  GEMMA3_4B_IT = 'gs://gemma-data/checkpoints/gemma3-4b-it'
  GEMMA3_12B_IT = 'gs://gemma-data/checkpoints/gemma3-12b-it'
  GEMMA3_27B_IT = 'gs://gemma-data/checkpoints/gemma3-27b-it'

  # ******** Gemma 3.0 N ********
  # Pretrained
  GEMMA3N_E2B_PT = 'gs://gemma-data/checkpoints/gemma3n-e2b-pt'
  GEMMA3N_E4B_PT = 'gs://gemma-data/checkpoints/gemma3n-e4b-pt'
  # Instruction Tuned
  GEMMA3N_E2B_IT = 'gs://gemma-data/checkpoints/gemma3n-e2b-it'
  GEMMA3N_E4B_IT = 'gs://gemma-data/checkpoints/gemma3n-e4b-it'
