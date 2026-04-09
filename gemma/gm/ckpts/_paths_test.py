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

"""Tests for checkpoint paths."""

import re

from gemma.gm.ckpts import _paths


class TestCheckpointPath:
  """Tests for CheckpointPath enum."""

  def test_is_str_enum(self):
    """All enum values should be strings."""
    for member in _paths.CheckpointPath:
      assert isinstance(member.value, str)

  def test_all_paths_start_with_gs(self):
    """All checkpoint paths should be GCS paths."""
    for member in _paths.CheckpointPath:
      assert member.value.startswith('gs://'), (
          f'{member.name} does not start with gs://: {member.value}'
      )

  def test_all_paths_contain_checkpoints(self):
    """All paths should contain the 'checkpoints' directory."""
    for member in _paths.CheckpointPath:
      assert '/checkpoints/' in member.value, (
          f'{member.name} missing /checkpoints/ in path: {member.value}'
      )

  def test_path_format_matches_name(self):
    """Path basename should be a lowercased version consistent with the name.

    E.g. GEMMA3_4B_IT -> gemma3-4b-it
    """
    for member in _paths.CheckpointPath:
      basename = member.value.rsplit('/', 1)[-1]
      # Convert name: GEMMA3_4B_IT -> gemma3-4b-it
      expected = member.name.lower().replace('_', '-')
      # Handle gemma3n enum names: GEMMA3N_E2B_IT -> gemma3n-e2b-it
      assert basename == expected, (
          f'{member.name}: expected basename {expected!r}, got {basename!r}'
      )

  def test_naming_convention_version_size_variant(self):
    """All names should follow VERSION_SIZE_VARIANT pattern."""
    pattern = re.compile(
        r'^GEMMA[23N]*_'  # Version: GEMMA2, GEMMA3, GEMMA3N
        r'(270M|1B|2B|4B|9B|12B|27B|E2B|E4B)_'  # Size
        r'(PT|IT)$'  # Variant: Pre-trained or Instruction-Tuned
    )
    for member in _paths.CheckpointPath:
      assert pattern.match(member.name), (
          f'{member.name} does not match VERSION_SIZE_VARIANT pattern'
      )

  def test_pt_and_it_pairs_exist(self):
    """Every model size should have both PT and IT variants."""
    names = [m.name for m in _paths.CheckpointPath]
    pt_names = {n.replace('_PT', '') for n in names if n.endswith('_PT')}
    it_names = {n.replace('_IT', '') for n in names if n.endswith('_IT')}
    assert pt_names == it_names, (
        f'PT/IT mismatch. PT-only: {pt_names - it_names}, '
        f'IT-only: {it_names - pt_names}'
    )

  def test_gemma2_models_exist(self):
    """Gemma 2 should have 2B, 9B, 27B sizes."""
    expected = {'GEMMA2_2B_PT', 'GEMMA2_9B_PT', 'GEMMA2_27B_PT',
                'GEMMA2_2B_IT', 'GEMMA2_9B_IT', 'GEMMA2_27B_IT'}
    names = {m.name for m in _paths.CheckpointPath}
    assert expected.issubset(names)

  def test_gemma3_models_exist(self):
    """Gemma 3 should have 270M, 1B, 4B, 12B, 27B sizes."""
    expected = {'GEMMA3_270M_PT', 'GEMMA3_1B_PT', 'GEMMA3_4B_PT',
                'GEMMA3_12B_PT', 'GEMMA3_27B_PT',
                'GEMMA3_270M_IT', 'GEMMA3_1B_IT', 'GEMMA3_4B_IT',
                'GEMMA3_12B_IT', 'GEMMA3_27B_IT'}
    names = {m.name for m in _paths.CheckpointPath}
    assert expected.issubset(names)

  def test_gemma3n_models_exist(self):
    """Gemma 3N should have E2B and E4B sizes."""
    expected = {'GEMMA3N_E2B_PT', 'GEMMA3N_E4B_PT',
                'GEMMA3N_E2B_IT', 'GEMMA3N_E4B_IT'}
    names = {m.name for m in _paths.CheckpointPath}
    assert expected.issubset(names)

  def test_str_enum_lookup(self):
    """Should be able to use string value to look up the enum."""
    path = 'gs://gemma-data/checkpoints/gemma3-4b-it'
    member = _paths.CheckpointPath(path)
    assert member is _paths.CheckpointPath.GEMMA3_4B_IT

  def test_no_duplicate_paths(self):
    """All paths should be unique."""
    values = [m.value for m in _paths.CheckpointPath]
    assert len(values) == len(set(values)), 'Duplicate checkpoint paths found'

  def test_total_count(self):
    """Verify the expected total number of checkpoints."""
    # Gemma2: 3 sizes * 2 variants = 6
    # Gemma3: 5 sizes * 2 variants = 10
    # Gemma3N: 2 sizes * 2 variants = 4
    # Total = 20
    assert len(_paths.CheckpointPath) == 20
