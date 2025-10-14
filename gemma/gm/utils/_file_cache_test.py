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

import os
import pathlib
from etils import epath
from gemma.gm.utils import _file_cache
import pytest


@pytest.fixture()
def cache_dir(tmp_path: pathlib.Path):
  os.environ['GEMMA_CACHE_DIR'] = os.fspath(tmp_path)
  yield tmp_path
  del os.environ['GEMMA_CACHE_DIR']


def test_cache_miss_returns_remote_path():
  if 'GEMMA_CACHE_DIR' in os.environ:
    del os.environ['GEMMA_CACHE_DIR']
  # As we do not create the cache file, there will be a miss and the remote path
  # will be returned.
  filename = _file_cache.maybe_get_from_cache(
      remote_file_path='/this/path/will/not/be/used.txt',
      cache_subdir='test',
  )
  assert filename == epath.Path('/this/path/will/not/be/used.txt')


def test_cache_dir_from_env_var(cache_dir: pathlib.Path):  # pylint: disable=redefined-outer-name
  dirpath = epath.Path(cache_dir) / 'test'
  expected_filename = dirpath / 'used.txt'

  # Create the cache directory and write a file in it.
  dirpath.mkdir()
  expected_filename.write_text('some text')

  filename = _file_cache.maybe_get_from_cache(
      remote_file_path='/this/path/will/not/be/used.txt',
      cache_subdir='test',
  )
  assert filename == expected_filename


def test_cache_in_default_location():
  if 'GEMMA_CACHE_DIR' in os.environ:
    del os.environ['GEMMA_CACHE_DIR']
  # We can't create a file in the default location so we test only the function
  # that returns the path.
  cache_path = _file_cache._get_cache_dir()
  assert cache_path == epath.Path('~/.gemma/').expanduser()
