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

import os
from etils import epath
from etils import epy
from gemma.gm.tools import _tools


class FileExplorer(_tools.Tool):
  """File explorer tool (read-only, sandboxed).

  This tool restricts file access to a sandbox directory to prevent arbitrary
  file reads. By default, it only allows access to the current working directory
  or a specified sandbox root.
  """

  DESCRIPTION = 'File explorer tool (read-only, sandboxed).'
  EXAMPLE = _tools.Example(
      query="What's the content of ./data/foo.txt?",
      thought='I need to `cat ./data/foo.txt`.',
      tool_kwargs={'method': 'cat', 'path': './data/foo.txt'},
      tool_kwargs_doc={'method': 'ONEOF(cat, ls)', 'path': '<PATH>'},
      result='...',
      answer='Here is the content of the file: ...',
  )
  KEYWORDS = ('file', 'directory', 'folder')

  def __init__(self, sandbox_root: str | epath.Path | None = None):
    """Initialize FileExplorer with optional sandbox restriction.

    Args:
      sandbox_root: Root directory for file access. If None, defaults to current
        working directory. All file operations are restricted to this directory
        and its subdirectories.
    """
    super().__init__()
    if sandbox_root is None:
      # Default to current working directory as sandbox
      sandbox_root = os.getcwd()
    self._sandbox_root = epath.Path(sandbox_root).resolve()
    # Ensure sandbox exists
    if not self._sandbox_root.exists():
      raise ValueError(f'Sandbox root does not exist: {self._sandbox_root}')

  def _validate_path(self, path: epath.Path) -> tuple[bool, str]:
    """Validates that the path is within the sandbox and not sensitive.

    Returns:
      Tuple of (is_valid, error_message). If is_valid is True, error_message is empty.
    """
    # Resolve to absolute path
    resolved_path = path.resolve()

    # Block access to sensitive system paths
    sensitive_prefixes = [
        '/etc',
        '/usr',
        '/bin',
        '/sbin',
        '/var',
        '/sys',
        '/proc',
        '/dev',
        '/root',
    ]
    # Also block home directory sensitive paths
    home = epath.Path(os.path.expanduser("~"))
    sensitive_home_paths = [
        home / '.ssh',
        home / '.bashrc',
        home / '.bash_history',
        home / '.zshrc',
        home / '.gitconfig',
        home / '.env',
    ]

    path_str = str(resolved_path)
    for prefix in sensitive_prefixes:
      if path_str.startswith(prefix):
        return False, f'Access denied: Path is in a restricted system directory ({prefix})'

    for sensitive_path in sensitive_home_paths:
      # Check if path equals or is a subpath of sensitive path
      try:
        resolved_path.relative_to(sensitive_path)
        return False, f'Access denied: Path is restricted for security ({sensitive_path})'
      except ValueError:
        # Path is not relative to sensitive_path, continue checking
        if resolved_path == sensitive_path:
          return False, f'Access denied: Path is restricted for security ({sensitive_path})'

    # Ensure path is within sandbox
    try:
      resolved_path.relative_to(self._sandbox_root)
    except ValueError:
      return False, (
          f'Access denied: Path is outside sandbox. '
          f'Sandbox root: {self._sandbox_root}, '
          f'Requested path: {resolved_path}'
      )

    return True, ''

  def call(self, method: str, path: str) -> str:  # pytype: disable=signature-mismatch
    """Reads file or lists directory (sandboxed).

    Args:
      method: Either 'cat' to read a file or 'ls' to list directory contents.
      path: Path to file or directory. Must be within the sandbox root.

    Returns:
      File contents or directory listing, or error message if access is denied.
    """
    path = epath.Path(path)

    # Validate path before any operation
    is_valid, error_msg = self._validate_path(path)
    if not is_valid:
      return error_msg

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
