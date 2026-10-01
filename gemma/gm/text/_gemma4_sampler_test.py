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

from gemma import gm
import pytest


class _Gemma3DummyTokenizer(gm.testing.DummyTokenizer):
  VERSION = 3


def test_rejects_incompatible_tokenizer():
  with pytest.raises(ValueError, match='Incompatible model and tokenizer'):
    gm.text.Gemma4Sampler(
        model=gm.nn.Gemma4_E2B(),
        params={},
        tokenizer=_Gemma3DummyTokenizer(),
    )
