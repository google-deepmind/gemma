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

"""Generate documentation.

Usage (from the root directory):

```sh
pip install -e .[docs]

sphinx-build -b html docs/ docs/_build
```
"""

import apitree


_COLABS_NAMES = [
    'tool_use',
    'finetuning',
    'lora_sampling',
    'lora_finetuning',
    'multimodal',
    'quantization_aware_training',
    'quantization_sampling',
    'sampling',
    'sharding',
    'tokenizer',
]


apitree.make_project(
    modules={
        'gm': 'gemma.gm',
        'peft': 'gemma.peft',
    },
    # TODO(epot): Support mkdir parent for the destination, to support
    # `'colab/finetuning.ipynb'` as output.
    includes_paths={
        f'colabs/{name}.ipynb': f'colab_{name}.ipynb' for name in _COLABS_NAMES
    }
    | {
        'gemma/peft/README.md': 'peft.md',
    },
    # Redirect the empty `XX.html` pages to their `colab_XX.html`
    redirects={name: f'colab_{name}.html' for name in _COLABS_NAMES},
    globals=globals(),
)
