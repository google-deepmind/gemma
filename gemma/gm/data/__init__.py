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

"""Data pipeline ops."""

from etils import epy as _epy

# pylint: disable=g-import-not-at-top

with _epy.lazy_api_imports(globals()):

  # pylint: disable=g-importing-member,g-bad-import-order

  # Data sources
  from gemma.gm.data._parquet import Parquet

  # `transforms=`
  from gemma.gm.data._tasks import ContrastiveTask
  from gemma.gm.data._tasks import Seq2SeqTask
  from gemma.gm.data._transforms import AddSeq2SeqFields
  from gemma.gm.data._transforms import DecodeBytes
  from gemma.gm.data._transforms import FormatText
  from gemma.gm.data._transforms import MapInts
  from gemma.gm.data._transforms import Pad
  from gemma.gm.data._transforms import Tokenize

  # Functional API
  from gemma.gm.data._functional import make_seq2seq_fields
  from gemma.gm.data._functional import pad
