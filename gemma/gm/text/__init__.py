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

"""Text processing utilities."""

from etils import epy as _epy

# pylint: disable=g-import-not-at-top,g-importing-member

with _epy.lazy_api_imports(globals()):
  # Tokenizers
  from gemma.gm.text._tokenizer import Gemma2Tokenizer
  from gemma.gm.text._tokenizer import Gemma3Tokenizer
  from gemma.gm.text._tokenizer import Gemma3nTokenizer
  from gemma.gm.text._tokenizer import Gemma4Tokenizer
  from gemma.gm.text._tokenizer import Tokenizer
  from gemma.gm.text._tokenizer import SpecialTokens

  # Samplers
  from gemma.gm.text._sampler import Sampler
  from gemma.gm.text._chat_sampler import ChatSampler
  from gemma.gm.text._gemma4_sampler import Gemma4Sampler
  from gemma.gm.text._tool_sampler import ToolSampler

  # Sampling methods
  # TODO(mblondel): Add nucleus sampling
  from gemma.gm.text._sampling import SamplingMethod
  from gemma.gm.text._sampling import Greedy
  from gemma.gm.text._sampling import RandomSampling
  from gemma.gm.text._sampling import TopkSampling
  from gemma.gm.text._sampling import TopPSampling

  # System One Sampler
  from gemma.gm.text._systemone_sampler import SystemOneSampler
  from gemma.gm.text._systemone_sampler import QuestionType
  from gemma.gm.text._systemone_sampler import QuestionSpec
  from gemma.gm.text._systemone_sampler import NoulResult
  from gemma.gm.text._systemone_sampler import ChoiceResult
  from gemma.gm.text._systemone_sampler import ScoreResult
  from gemma.gm.text._systemone_sampler import DecisionResult
  from gemma.gm.text._systemone_sampler import SystemOneResponse
  from gemma.gm.text._systemone_sampler import build_tree_attention_pack
  from gemma.gm.text._systemone_sampler import format_noul_prompt
  from gemma.gm.text._systemone_sampler import format_choice_prompt
  from gemma.gm.text._systemone_sampler import format_score_prompt

  # System One Debiasing & Calibration
  from gemma.gm.text._systemone_debias import calibrate_and_score
  from gemma.gm.text._systemone_debias import spread_order
  from gemma.gm.text._systemone_debias import cyclic_shifts
  from gemma.gm.text._systemone_debias import marginalize_cyclic_distributions
  from gemma.gm.text._systemone_debias import compute_order_flip_rate
  from gemma.gm.text._systemone_debias import TemperatureScaler
  from gemma.gm.text._systemone_debias import fit_temperature
  from gemma.gm.text._systemone_debias import compute_ece

  # Other utils
  # from gemma.gm.text import _template as template
