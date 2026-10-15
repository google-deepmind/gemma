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

"""Reference conformance fixtures for Gemma implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DEFAULT_FIXTURE = Path(__file__).with_name('v1') / 'gemma2_tokenizer.json'


def load_fixture(path: str | Path | None = None) -> dict[str, Any]:
  """Loads a versioned conformance fixture."""
  fixture_path = Path(path) if path is not None else _DEFAULT_FIXTURE
  return json.loads(fixture_path.read_text(encoding='utf-8'))


def check_tokenizer(tokenizer: Any, fixture: dict[str, Any]) -> list[str]:
  """Returns mismatch descriptions for a tokenizer fixture."""
  mismatches = []
  for case in fixture['cases']:
    actual_ids = tokenizer.encode(
        case['text'],
        add_bos=case.get('add_bos', False),
        add_eos=case.get('add_eos', False),
    )
    if actual_ids != case['token_ids']:
      mismatches.append(
          f"{case['name']}: token ids {actual_ids} != {case['token_ids']}"
      )
      continue

    expected_text = case.get('decoded_text')
    if expected_text is not None:
      actual_text = tokenizer.decode(case['token_ids'])
      if actual_text != expected_text:
        mismatches.append(
            f"{case['name']}: decoded text {actual_text!r} != {expected_text!r}"
        )
  return mismatches
