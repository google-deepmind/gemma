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

from gemma import conformance


class _FakeTokenizer:

  def encode(self, text, *, add_bos=False, add_eos=False):
    ids = [len(text)]
    if add_bos:
      ids.insert(0, 2)
    if add_eos:
      ids.append(1)
    return ids

  def decode(self, ids):
    del ids
    return 'decoded'


def test_check_tokenizer_passes_matching_case():
  fixture = {
      'cases': [{
          'name': 'example',
          'text': 'abc',
          'add_bos': True,
          'add_eos': True,
          'token_ids': [2, 3, 1],
          'decoded_text': 'decoded',
      }]
  }

  assert not conformance.check_tokenizer(_FakeTokenizer(), fixture)


def test_check_tokenizer_reports_token_id_mismatch():
  fixture = {
      'cases': [{
          'name': 'example',
          'text': 'abc',
          'token_ids': [99],
      }]
  }

  mismatches = conformance.check_tokenizer(_FakeTokenizer(), fixture)

  assert mismatches == ['example: token ids [3] != [99]']


def test_default_fixture_is_versioned_gemma2_tokenizer():
  fixture = conformance.load_fixture()

  assert fixture['schema_version'] == 1
  assert fixture['kind'] == 'tokenizer'
  assert fixture['tokenizer']['version'] == 2
  assert fixture['cases']
