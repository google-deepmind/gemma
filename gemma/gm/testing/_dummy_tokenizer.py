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

"""A dummy tokenizer for testing."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import functools
import typing

from etils import epath
from gemma.gm.text import _tokenizer

import sentencepiece as spm

if typing.TYPE_CHECKING:
  _base_cls = (spm.SentencePieceProcessor,)
else:
  _base_cls = ()


class _DummySentencePieceProcessor(*_base_cls):
  """Dummy tokenizer."""

  def __init__(self):
    super().__init__()
    self._start_id = 3
    self._mapping_text_to_id = {
        '<pad>': 0,
        '<s>': 1,
        '</s>': 2,
        'input': 3,
        'string': 4,
        'hello': 5,
        'world': 6,
        'Hello': 7,
        'there': 8,
        '!': 9,
        'My': 10,
        'name': 11,
        'is': 12,
        'Morgane': 13,
    }
    self._vocab_size = len(self._mapping_text_to_id)

  def pad_id(self) -> int:
    return 0

  def bos_id(self) -> int:
    return 1

  def eos_id(self) -> int:
    return 2

  def GetPieceSize(self) -> int:  # pylint: disable=invalid-name
    return self._vocab_size

  def DecodeIds(self, ids: Iterable[int]) -> str:  # pylint: disable=invalid-name
    reverse_mapping = {v: k for k, v in self._mapping_text_to_id.items()}
    return ' '.join(reverse_mapping[e] for e in ids)

  def EncodeAsIds(self, text: str) -> list[int]:  # pylint: disable=invalid-name
    words = text.split(' ')
    return [self._mapping_text_to_id[word] for word in words]


@dataclasses.dataclass(frozen=True, kw_only=True)
class DummyTokenizer(_tokenizer.Tokenizer):
  """Dummy tokenizer."""

  path: epath.PathLike = '/tmp/dummy_tokenizer.model'

  special_tokens = _tokenizer._Gemma3SpecialTokens  # pylint: disable=protected-access

  @functools.cached_property
  def _sp(self) -> spm.SentencePieceProcessor:
    return _DummySentencePieceProcessor()
