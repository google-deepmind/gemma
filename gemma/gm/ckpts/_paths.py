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

"""Checkpoint paths."""

import enum


class CheckpointPath(enum.StrEnum):
  """Hardcoded paths to Gemma checkpoints.

  Format: `{VERSION}_{SIZE}_{VARIANT}`.

  * `PT`: Pre-trained
  * `IT`: Instruction Tuned

  For example, `GEMMA2_27B_IT` is Gemma V2, 27 Billion parameters, instruction
  tuned.
  """

  # TODO(epot): Add other versions.
  GEMMA2_2B_PT = 'gs://gemma-data/checkpoints/gemma2-2b-pt/'
  GEMMA2_2B_IT = 'gs://gemma-data/checkpoints/gemma2-2b-it/'
