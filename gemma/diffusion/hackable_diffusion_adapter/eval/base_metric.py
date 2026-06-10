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

"""Base classes for host-side detokenizing metrics."""

from __future__ import annotations

import abc
import dataclasses
import functools
import flax.struct
import jax.numpy as jnp
from kauldron import kd
import numpy as np
import seqio

_GEMMA4_TOKENIZER_PATH = (
    "gs://gemma-data/tokenizers/tokenizer_gemma4.model"
)


@dataclasses.dataclass(kw_only=True, frozen=True)
class BaseTokenizerMetric(kd.metrics.Metric, abc.ABC):
  """Base class for metrics requiring host-side tokenizer access."""

  tokenizer_path: str = _GEMMA4_TOKENIZER_PATH

  @functools.cached_property
  def tokenizer(self) -> seqio.vocabularies.SentencePieceVocabulary:
    return seqio.vocabularies.SentencePieceVocabulary(self.tokenizer_path)

  def _squeeze_3d(self, x: jnp.ndarray) -> jnp.ndarray:
    """Squeezes 3D [B, L, 1] arrays to 2D [B, L]."""
    if x.ndim == 3:
      return x[..., 0]
    return x

  def decode_batch(self, tokens_np: np.ndarray) -> list[str]:
    """Decodes 2D token array to a list of strings on host."""
    return [self.tokenizer.decode(row.tolist()) for row in tokens_np]


@dataclasses.dataclass(kw_only=True, frozen=True)
class BaseTextMetric(BaseTokenizerMetric):
  """Base class for metrics requiring host-side detokenization."""

  tokens: kd.kontext.Key = kd.kontext.REQUIRED
  ground_truth: kd.kontext.Key = kd.kontext.REQUIRED


@dataclasses.dataclass(kw_only=True, frozen=True)
class BaseSimpleTextMetric(BaseTextMetric):
  """Metric for simple text tasks returning a single value per example.

  Stores raw token arrays in the metric state and performs all
  detokenization and scoring on the host in ``compute()``.
  """

  @flax.struct.dataclass
  class State(kd.metrics.AutoState["BaseSimpleTextMetric"]):
    """Collects raw tokens across batches for host-side scoring."""

    tokens: jnp.ndarray = kd.metrics.concat_field()
    ground_truth: jnp.ndarray = kd.metrics.concat_field()

    def compute(self) -> float:
      """Detokenize and score all accumulated examples on host."""
      final = self.finalize()
      tokens_np = np.asarray(final.tokens)
      gt_np = np.asarray(final.ground_truth)

      gen_texts = self.parent.decode_batch(tokens_np)
      gt_texts = self.parent.decode_batch(gt_np)

      num_examples = len(gen_texts)
      if num_examples == 0:
        return 0.0
      total = sum(
          float(self.parent.score_example(gen_texts[i], gt_texts[i]))
          for i in range(num_examples)
      )
      return total / num_examples

  @abc.abstractmethod
  def score_example(
      self, generated_text: str, ground_truth_text: str
  ) -> float | bool:
    """Scores a single example."""
    pass

  def get_state(
      self,
      *,
      tokens: jnp.ndarray,
      ground_truth: jnp.ndarray,
  ) -> BaseSimpleTextMetric.State:
    """Store raw tokens — all scoring is deferred to ``compute()``."""
    return self.State(
        tokens=self._squeeze_3d(tokens),
        ground_truth=self._squeeze_3d(ground_truth),
    )
