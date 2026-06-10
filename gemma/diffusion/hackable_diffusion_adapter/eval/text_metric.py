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

"""Detokenization metric for logging prompt-response text."""

from __future__ import annotations

import dataclasses

import flax.struct
from gemma.diffusion.hackable_diffusion_adapter.eval import base_metric
from kauldron import kd
from kauldron.ktyping import Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member

################################################################################
# MARK: DetokenizePromptAndResponse
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class DetokenizePromptAndResponse(base_metric.BaseTokenizerMetric):
  """Detokenize prompt and response tokens, log them as combined text."""

  prompt: kd.kontext.Key = kd.kontext.REQUIRED
  response: kd.kontext.Key = kd.kontext.REQUIRED

  num_texts: int = 5
  separator: str = ""

  @flax.struct.dataclass
  class State(kd.metrics.AutoState["DetokenizePromptAndResponse"]):
    """Collects the first num_texts prompt+response pairs."""

    prompt: Int["b p"] = kd.metrics.truncate_field(num_field="parent.num_texts")
    response: Int["b r"] = kd.metrics.truncate_field(
        num_field="parent.num_texts"
    )

    def compute(self) -> list[str]:
      """Detokenizes collected prompts and responses, pairing them together."""
      results = []
      for p, r in zip(self.prompt, self.response):
        prompt_text = self.parent.tokenizer.decode(p)
        response_text = self.parent.tokenizer.decode(r)
        results.append(prompt_text + self.parent.separator + response_text)
      return results

  @typechecked
  def get_state(
      self,
      *,
      prompt: Int["b p ..."],
      response: Int["b r ..."],
  ) -> DetokenizePromptAndResponse.State:
    """Collects prompt and response tokens into the metric state."""
    # Squeeze trailing singleton dim if present.
    prompt = self._squeeze_3d(prompt)
    response = self._squeeze_3d(response)
    return self.State(prompt=prompt, response=response)
