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

import pickle
import pytest

from gemma import gm

# Activate the fixture
use_hermetic_tokenizer = gm.testing.use_hermetic_tokenizer


def test_pickle():
  tokenizer = gm.text.Gemma3Tokenizer()
  tokenizer.encode('Hello world!')  # Trigger the lazy-loading of the tokenizer.

  pickle.dumps(tokenizer)


def test_encode_invalid_inputs():
  tokenizer = gm.text.Gemma3Tokenizer()

  with pytest.raises(TypeError, match='tokenizer.encode expects str or list'):
    tokenizer.encode(123)

  with pytest.raises(TypeError, match='tokenizer.encode expects str or list'):
    tokenizer.encode(None)
