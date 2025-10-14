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

"""Utility for caching file on file system."""

import os
from etils import epath

# If defined the tokenizer model will be fetched from this local directory
# instead of being downloaded.
_GEMMA_CACHE_DIR_ENV_NAME = 'GEMMA_CACHE_DIR'
_DEFAULT_CACHE_DIR = '~/.gemma'


def maybe_get_from_cache(
    *,
    remote_file_path: epath.PathLike,
    cache_subdir: str,
) -> epath.Path:
  """Returns the cached file if exists, otherwise returns the remote file path."""
  filename = epath.Path(remote_file_path).name

  cache_filepath = _get_cache_dir() / cache_subdir / filename
  if cache_filepath.exists():
    return cache_filepath
  return epath.Path(remote_file_path)


def _get_cache_dir() -> epath.Path:
  """Get the path to the cached file."""
  cache_dir = os.environ.get(_GEMMA_CACHE_DIR_ENV_NAME, _DEFAULT_CACHE_DIR)
  return epath.Path(cache_dir).expanduser()
