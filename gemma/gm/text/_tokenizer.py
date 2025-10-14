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

"""Tokenizer API."""

from __future__ import annotations

import dataclasses
import enum
import functools
import typing
from typing import ClassVar

import einops
from etils import enp
from etils import epath
from etils import epy
from gemma.gm.utils import _file_cache
import jax
import jax.numpy as jnp
from kauldron.utils import immutabledict
import numpy as np

from sentencepiece import sentencepiece_model_pb2
import sentencepiece as spm

with epy.lazy_imports():
  from plotly import graph_objects as go  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  import plotly.express as px  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error


_WHITESPACE_CHAR = '▁'  # Note this is NOT a undescore (▁ != _)


class _DisplayEnumType(enum.EnumType):
  """Metaclass which displays the enum values when the class is printed."""

  def __repr__(cls):
    return epy.Lines.make_block(
        header=cls.__name__,
        content={value.name: value.value for value in cls},
    )


class SpecialTokens(enum.IntEnum, metaclass=_DisplayEnumType):
  """Special tokens ids."""

  PAD: ClassVar[int]
  EOS: ClassVar[int]
  BOS: ClassVar[int]
  UNK: ClassVar[int]
  MASK: ClassVar[int]

  # Initial index to access the `<unusedXX>` tokens. For example, `<unused7>` is
  # `SpecialTokens.CUSTOM + 7`
  CUSTOM: ClassVar[int]
  START_OF_TURN: ClassVar[int]  # <start_of_turn>
  END_OF_TURN: ClassVar[int]  # <end_of_turn>

  START_OF_IMAGE: ClassVar[int]  # '<start_of_image>'
  END_OF_IMAGE: ClassVar[int]  # '<end_of_image>'


class _Gemma2SpecialTokens(SpecialTokens, enum.IntEnum):
  """Special tokens ids."""

  PAD = 0
  EOS = 1
  BOS = 2
  UNK = 3
  MASK = 4
  # '<2mass>' = 5
  # '[@BOS@]' = 6
  # '[multimodal]' = 7
  # Initial index to access the `<unusedXX>` tokens. For example, `<unused7>` is
  # `SpecialTokens.CUSTOM + 7`
  CUSTOM = 7
  # <unused1> = 8
  # <unused2> = 9
  # ...
  START_OF_TURN = 106  # <start_of_turn>
  END_OF_TURN = 107  # <end_of_turn>

  # '<start_of_image>' = 255999


class _Gemma3SpecialTokens(SpecialTokens, enum.IntEnum):
  """Special tokens ids."""

  PAD = 0
  EOS = 1
  BOS = 2
  UNK = 3
  MASK = 4
  # '[multimodal]' = 5
  # Initial index to access the `<unusedXX>` tokens. For example, `<unused7>` is
  # `SpecialTokens.CUSTOM + 7`
  CUSTOM = 6  # <unused0>
  # <unused1> = 7
  # <unused2> = 8
  # ...
  # TODO(epot): Tokenizer also has `<unused99>` up to `<unused6238>` after the
  # `<START_OF_IMAGE>` token (starting at 256000).
  START_OF_TURN = 105  # <start_of_turn>
  END_OF_TURN = 106  # <end_of_turn>

  # Multimodal tokens (Gemma3 only)
  START_OF_IMAGE = 255999  # '<start_of_image>'
  END_OF_IMAGE = 256000  # <end_of_image>


@dataclasses.dataclass(frozen=True, kw_only=True)
class Tokenizer:
  """Base class for tokenizers.

  ```python
  tokenizer = gm.text.Gemma2Tokenizer()

  tokenizer.encode('Hello world!')
  tokenizer.decode([10, 20, 30, 40, 50])

  print(tokenizer.tokens[:200])  # Print the first 200 tokens.

  assert (
      tokenizer.tokens[tokenizer.special_tokens.START_OF_TURN]
      == '<start_of_turn>'
  )
  ```

  Attributes:
    path: Path to the vocab file.
    custom_tokens: The Gemma tokenizer has a few unused tokens which can be
      overwritten by the user here. Expect a dictionary mapping the unused id
      (0-98) to the token string. (`e.g. `{0: '<start_of_audio>'}`)
    VERSION: The Gemma version of the tokenizer (2, 3).
    FORBIDDEN_TOKENS: Default forbidden tokens.
  """

  path: epath.PathLike
  custom_tokens: dict[int, str] = dataclasses.field(default_factory=dict)

  VERSION: ClassVar[int | str] = 0
  FORBIDDEN_TOKENS: ClassVar[tuple[int, ...]] = ()  # pylint: disable=g-missing-from-attributes

  def __post_init__(self):
    immutabledict.freeze_dict_attrs(self, ['custom_tokens'])

  @classmethod
  def from_version(cls, version: int | str) -> Tokenizer:
    """Create a tokenizer from a version."""
    if version == 2:
      return Gemma2Tokenizer()
    elif version == 3:
      return Gemma3Tokenizer()
    elif version == '3n':
      return Gemma3nTokenizer()
    else:
      raise ValueError(f'Unsupported tokenizer version: {version}')

  def encode(
      self,
      text: str | list[str],
      *,
      add_bos: bool = False,
      add_eos: bool = False,
  ) -> list[int]:
    """Encode a text into a list of token ids.

    ```python
    tokenizer = gm.text.Gemma2Tokenizer()
    tokenizer.encode('Hello world!')

    pieces = tokenizer.split('Hello world!')
    assert pieces == ['Hello', ' world', '!']
    tokenizer.encode(pieces)
    ```

    Args:
      text: The text to encode. Can be a single string or a list of tokens.
      add_bos: Whether to prepend the BOS token (`2`) (begin of sentence).
      add_eos: Whether to append the EOS token (`1`) (end of sentence).

    Returns:
      The list of token ids.
    """
    if isinstance(text, str):
      token_ids = self._sp.EncodeAsIds(text)
    else:
      token_ids = [
          self._sp.PieceToId(t.replace(' ', _WHITESPACE_CHAR)) for t in text
      ]
      if self.special_tokens.UNK in token_ids:
        index = token_ids.index(self.special_tokens.UNK)
        raise ValueError(
            f'Cannot tokenize {text!r}. Token {text[index]!r} is an unknown'
            ' token.'
        )

    if add_bos:
      token_ids.insert(0, self.special_tokens.BOS)
    if add_eos:
      token_ids.append(self.special_tokens.EOS)
    return token_ids

  def decode(self, ids: int | list[int] | enp.typing.Array) -> str:
    if isinstance(ids, int):
      ids = [ids]
    elif enp.lazy.is_array(ids):  # Supports decoding from jnp, np arrays
      ids = typing.cast(np.ndarray, ids)
      if ids.ndim == 0:  # scalar
        ids = [ids.item()]
      elif ids.ndim == 1:
        ids = ids.tolist()
      else:
        raise ValueError(f'Array must be 0 or 1 dimensional, got {ids.shape}.')
    return self._sp.DecodeIds(ids)

  def split(self, text: str) -> list[str]:
    """Split a text into pieces."""
    return [_real_whitespaces(t) for t in self._sp.EncodeAsPieces(text)]

  @functools.cached_property
  def vocab_size(self) -> int:
    """Size of the vocabulary."""
    return self._sp.GetPieceSize()

  @functools.cached_property
  def tokens(self) -> list[str]:
    """Returns the list of all tokens `str` from the vocabulary."""
    return [
        _real_whitespaces(self._sp.IdToPiece(i)) for i in range(self.vocab_size)
    ]

  @functools.cached_property
  def special_tokens(self) -> type[SpecialTokens]:
    """Returns the special tokens."""
    raise NotImplementedError(
        f'{type(self).__qualname__} does not define special tokens.'
    )

  # TODO(epot): Global cache so all instances do not reload the tokenizer.
  @functools.cached_property
  def _sp(self) -> spm.SentencePieceProcessor:
    """Returns the sentencepiece processor."""
    sp = spm.SentencePieceProcessor()
    model_file_path = _file_cache.maybe_get_from_cache(
        remote_file_path=self.path,
        cache_subdir='tokenizer',
    )
    model_proto = epath.Path(model_file_path).read_bytes()

    if self.custom_tokens:
      model_proto = self._add_custom_tokens(model_proto)

    sp.LoadFromSerializedProto(model_proto)
    return sp

  def _add_custom_tokens(self, serialized_proto: bytes) -> bytes:
    """Update the custom tokens of the proto."""
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(serialized_proto)

    for i, token in self.custom_tokens.items():
      if i < 0 or i > 98:
        raise ValueError(
            f'Custom token id {i} for {token!r} is not in [1, 98].'
        )

      # Update the piece
      piece = proto.pieces[self.special_tokens.CUSTOM + i]
      if piece.piece != f'<unused{i}>':
        raise AssertionError(
            f'Expected custom token id {i} for {token!r} to be `<unused{i}>`,'
            f' but was {piece.piece}. This indicates the voab file'
            " isn't as expected."
        )
      piece.piece = token

      # Update the user_defined_symbols
      # The user_defined_symbols do not have the same ids as the pieces.
      # Gemma3 tokenizer do not have `user_defined_symbols`.
      if proto.trainer_spec.user_defined_symbols:
        for index, symbol in enumerate(proto.trainer_spec.user_defined_symbols):
          if symbol == f'<unused{i}>':
            break
        else:
          raise AssertionError(
              f'Expected custom token id {i} for {token!r} to be in'
              ' user_defined_symbols, but it was not found.'
          )
        proto.trainer_spec.user_defined_symbols[index] = token
    return proto.SerializeToString()

  def plot_logits(
      self,
      logits: enp.typing.Array,
      *,
      keep_top: int = 30,
  ) -> go.Figure:
    """Plot the distribution of logits.

    Args:
      logits: The logits to plot, before softmax is applied (as returned by the
        model).
      keep_top: Number of tokens to display.

    Returns:
      The plot as a plotly figure.
    """
    if logits.ndim == 2 and logits.shape[0] == 1:  # batch_size==1, flatten
      logits = einops.rearrange(logits, '1 d -> d')
    if logits.ndim != 1:
      raise ValueError(
          '`plot_logits` expects logits for a single token, got'
          f' {logits.shape}.'
      )

    # Compute the probability distribution.
    probs = jax.nn.softmax(logits)

    # Select the top `keep_top` tokens.
    indices = jnp.argsort(probs)
    indices = indices[-keep_top:][::-1]

    # Plot the distribution.
    probs = probs[indices].astype(np.float32)
    words = [repr(self.tokens[i]) for i in indices]

    fig = px.bar(x=words, y=probs)

    # Customize the plot
    fig.update_layout(
        title='Probability Distribution of Tokens',
        xaxis_title='Tokens',
        yaxis_title='Probability',
    )
    return fig

  # Pickle protocol
  # The `spm.SentencePieceProcessor` is not pickleable, so instead implement
  # pickle protocol.

  def __getstate__(self):
    return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

  def __setstate__(self, state):
    self.__init__(**state)


@dataclasses.dataclass(frozen=True)
class Gemma2Tokenizer(Tokenizer):
  """Tokenizer for Gemma 2."""

  # TODO(epot): Add a util to auto-download and cache the tokenizer from gs://
  # bucket (e.g. in `~/.gemma/<tokenizer_name>`). Could be customized
  # through some `GEMMA_CACHE_DIR` environment variable.
  path: epath.PathLike = (
      'gs://gemma-data/tokenizers/tokenizer_gemma2.model'
  )

  special_tokens = _Gemma2SpecialTokens

  VERSION = 2


@dataclasses.dataclass(frozen=True)
class Gemma3Tokenizer(Tokenizer):
  """Tokenizer for Gemma 3."""

  # TODO(epot): Add a util to auto-download and cache the tokenizer from gs://
  # bucket (e.g. in `~/.gemma/<tokenizer_name>`). Could be customized
  # through some `GEMMA_CACHE_DIR` environment variable.
  # TODO(epot): Public GCS path
  path: epath.PathLike = (
      'gs://gemma-data/tokenizers/tokenizer_gemma3.model'
  )

  special_tokens = _Gemma3SpecialTokens

  # Tokens which are forbidden to be generated in the sampler.
  FORBIDDEN_TOKENS = (
      special_tokens.START_OF_IMAGE,
      special_tokens.END_OF_IMAGE,
  )

  VERSION = 3


@dataclasses.dataclass(frozen=True)
class Gemma3nTokenizer(Tokenizer):
  """Tokenizer for Gemma3n."""

  # TODO(epot): Add a util to auto-download and cache the tokenizer from gs://
  # bucket (e.g. in `~/.gemma/<tokenizer_name>`). Could be customized
  # through some `GEMMA_CACHE_DIR` environment variable.
  # TODO(epot): Public GCS path
  path: epath.PathLike = (
      'gs://gemma-data/tokenizers/tokenizer_gemma3n.model'
  )

  special_tokens = _Gemma3SpecialTokens

  # Tokens which are forbidden to be generated in the sampler.
  FORBIDDEN_TOKENS = (
      special_tokens.START_OF_IMAGE,
      special_tokens.END_OF_IMAGE,
  )

  VERSION = '3n'


def _real_whitespaces(text: str) -> str:
  """Normalize whitespaces."""
  return text.replace(_WHITESPACE_CHAR, ' ')
