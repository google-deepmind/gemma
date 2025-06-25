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

"""File explorer tool."""

from __future__ import annotations

from etils import epath
from etils import epy
from gemma.gm.tools import _tools


class FileExplorer(_tools.Tool):
  """File explorer tool (read-only)."""

  DESCRIPTION = 'File explorer tool.'
  EXAMPLE = _tools.Example(
      query="What's the content of /tmp/foo.txt?",
      thought='I need to `cat /tmp/foo.txt`.',
      tool_kwargs={'method': 'cat', 'path': '/tmp/foo.txt'},
      tool_kwargs_doc={'method': 'ONEOF(cat, ls)', 'path': '<PATH>'},
      result='...',
      answer='Here is the content of the file: ...',
  )
  KEYWORDS = ('file', 'directory', 'folder')

  def call(self, method: str, path: str) -> str:  # pytype: disable=signature-mismatch
    """Calculates the expression."""
    path = epath.Path(path)
    if method == 'cat':
      try:
        return path.read_text()
      except FileNotFoundError:
        return 'File not found. Make sure to use the absolute path.'
      except OSError as e:  # Trying to read a directory.
        return repr(e)
    elif method == 'ls':
      lines = epy.Lines()
      try:
        for p in path.iterdir():
          lines += f'{p.name}'
      except OSError as e:
        return repr(e)
      return lines.join()
    else:
      return f'Unknown method: {method}'
