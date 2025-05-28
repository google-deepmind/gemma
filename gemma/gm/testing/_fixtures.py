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

"""Tests fixtures."""

import contextlib  # pylint: disable=unused-import
from unittest import mock

from etils import epath
from gemma.gm.text import _tokenizer
import pytest  # pytype: disable=import-error


@pytest.fixture(autouse=True, scope='module')
def use_hermetic_tokenizer():
  """Use the local tokenizer, to avoid TFHub calls."""

  new_path = epath.resource_path('gemma') / 'testdata/tokenizer_gemma3.model'

  # We cannot mock `Gemma3Tokenizer.path` directly as dataclasses also
  # set the value in the `__init__` default value.

  old_init = _tokenizer.Gemma3Tokenizer.__init__

  def mew_init(self, path=None, **kwargs):
    del path
    old_init(self, new_path, **kwargs)

  with (
      contextlib.nullcontext()
  ):
    yield
