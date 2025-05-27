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

"""Utils for LoRA checkpoint managment."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, TypeVar, Union

from gemma import peft
from kauldron import kd

# Nested dict of params
_ParamsDict = Any | dict[str, Union['_ParamsDict', Any]]

if typing.TYPE_CHECKING:
  # Likely overkill, but avoid resolving the lazy-import on importing this file.
  _StateT = TypeVar('_StateT', bound=kd.train.TrainState)
else:
  _StateT = TypeVar('_StateT')


@dataclasses.dataclass(frozen=True)
class SkipLoRA(kd.ckpts.AbstractPartialLoader):
  """Wraps a partial loader to not restore the LoRA weights."""

  wrapped: kd.ckpts.AbstractPartialLoader

  def transform(self, state: _StateT) -> _StateT:  # pytype: disable=signature-mismatch
    # Remove the LoRA weights from the params structure so it can be restored
    original_params, lora_params = peft.split_params(state.params)

    state = state.replace(params=original_params)

    state = self.wrapped.transform(state)

    # Restore the LoRA weights
    state = state.replace(params=peft.merge_params(state.params, lora_params))

    return state
