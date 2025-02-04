# Copyright 2024 DeepMind Technologies Limited.
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


apitree.make_project(
    modules={
        'gm': 'gemma.gm',
        'peft': 'gemma.peft',
    },
    # TODO(epot): Support mkdir parent for the destination, to support
    # `'colab/finetuning.ipynb'` as output.
    includes_paths={
        'colabs/finetuning.ipynb': 'colab_finetuning.ipynb',
        'colabs/lora.ipynb': 'colab_lora.ipynb',
        'colabs/sampling.ipynb': 'colab_sampling.ipynb',
        'colabs/tokenizer.ipynb': 'colab_tokenizer.ipynb',
        'gemma/peft/README.md': 'peft.md',
    },
    globals=globals(),
)
