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

"""Tools."""

# pylint: disable=g-import-not-at-top,g-importing-member,g-bad-import-order

# Tool handler
from gemma.gm.tools._manager import ToolManagerBase
from gemma.gm.tools._manager import OneShotToolManager

# API to build new tools
from gemma.gm.tools._tools import Tool
from gemma.gm.tools._tools import ToolOutput
from gemma.gm.tools._tools import Example

# Available tools (mostly for demo purposes)
from gemma.gm.tools._calculator import Calculator
from gemma.gm.tools._file_explorer import FileExplorer
from gemma.gm.tools._offline_tool_search import OfflineToolSearch
