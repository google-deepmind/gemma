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

"""Command-line runner for Gemma conformance fixtures."""

from __future__ import annotations

import argparse

from gemma import conformance


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--fixture', help='Path to a tokenizer fixture JSON file.'
  )
  parser.add_argument(
      '--tokenizer-path',
      help=(
          'Tokenizer model path. Defaults to the source recorded in the'
          ' fixture.'
      ),
  )
  args = parser.parse_args()

  fixture = conformance.load_fixture(args.fixture)
  if fixture['kind'] != 'tokenizer' or fixture['tokenizer']['version'] != 2:
    parser.error(
        'This runner currently supports Gemma 2 tokenizer fixtures only.'
    )

  from gemma import gm  # Imported lazily to keep fixture inspection lightweight.

  tokenizer_path = args.tokenizer_path or fixture['tokenizer']['source']
  tokenizer = gm.text.Gemma2Tokenizer(path=tokenizer_path)
  mismatches = conformance.check_tokenizer(tokenizer, fixture)
  if mismatches:
    for mismatch in mismatches:
      print(f'FAIL: {mismatch}')
    return 1

  print(f"PASS: {len(fixture['cases'])} tokenizer conformance cases")
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
