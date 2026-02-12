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

"""File explorer tool."""

from __future__ import annotations

from typing import Annotated, Literal

from etils import epath
from etils import epy
import pydantic


def file_explorer(
    *,
    method: Annotated[
        Literal['cat', 'ls', 'exists'],
        pydantic.Field(description='The method to call.'),
    ],
    path: Annotated[
        str,
        pydantic.Field(
            description=(
                'The path to read. Supports remote filesystems, like `gs://`.'
            )
        ),
    ],
):  # pytype: disable=signature-mismatch
  """File explorer tool."""
  path = epath.Path(path)
  match method:
    case 'cat':
      return path.read_text()
    case 'ls':
      lines = epy.Lines()
      for p in path.iterdir():
        lines += f'{p.name}/' if p.is_dir() else p.name
      return lines.join()
    case 'exists':
      return path.exists()
    case _:
      raise ValueError(f'Unknown method: {method}')
