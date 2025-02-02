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

"""Sampling evaluator."""

from __future__ import annotations

import functools
from typing import Any, Optional

from flax import linen as nn
from gemma.gm.text import _sampler
from gemma.gm.text import _tokenizer
import jax
from kauldron import kd
import kauldron.data.utils as data_utils
from kauldron.utils import config_util


class SamplerEvaluator(kd.evals.EvaluatorBase):
  """Sampling evaluator.

  The evaluator expects

  Attributes:
    max_new_tokens: Maximum number of new tokens to generate. In total, the
      model will process `input_length + max_new_tokens`.
    tokenizer: The tokenizer to use.
    num_examples: How many examples to sample.
    ds: Dataset to evaluate on. Note that the dataset must be unbatched and
      contain raw `str` fields.
    prompt: Key of the prompt in the dataset.
    response: Key of the response in the dataset. Only used to write the ground
      truth to TensorBoard.
    model: The model to use.
  """

  # Sampler parameters

  max_new_tokens: int
  # TODO(epot): Auto-infer the tokenizer from the model ?
  tokenizer: _tokenizer.Tokenizer

  # Dataset parameters

  num_examples: Optional[int] = 1
  ds: kd.data.Pipeline = config_util.ROOT_CFG_REF.eval_ds
  prompt: kd.kontext.Key
  response: kd.kontext.Key | None = None

  model: nn.Module = config_util.ROOT_CFG_REF.model

  def evaluate(self, state: kd.train.TrainState, step: int) -> Any:
    """Run this evaluator then write and optionally return the results."""
    self._assert_root_cfg_resolved()

    sampler = _sampler.Sampler(
        model=self.base_cfg.model,
        params=state.params,
        tokenizer=self.tokenizer,
    )

    prompts = [kd.kontext.get_by_path(ex, self.prompt) for ex in self.examples]

    # TODO(epot): Supports batch_size for sampling.
    # Evaluators are run within the `jax.transfer_guard('disallow')` for the
    # train loop, so re-enable it here.
    with jax.transfer_guard('allow'):
      samples = sampler.sample(prompts, max_new_tokens=self.max_new_tokens)

    # TODO(epot): Would be nice to also write the samples to some text file,
    # Like `self.eval_path(step) / f'{self.name}_preds.txt'` or
    # `self.writer.path_for_step(step) ?`
    # TODO(epot): If that's the case, also create a XManager artifact ?
    texts = {}
    for i, (prompt, ex, sample) in enumerate(
        zip(prompts, self.examples, samples)
    ):
      texts[f'prompt_{i}'] = prompt
      texts[f'sample_{i}'] = sample
      if self.response:
        texts[f'gt_{i}'] = kd.kontext.get_by_path(ex, self.response)
    self.writer.write_texts(step, texts)

  @functools.cached_property
  def examples(self) -> list[Any]:
    """Extract the prompts from the dataset."""
    return list(self.ds.take(self.num_examples))


def _get_input_tokens(model, batch):
  args, kwargs = data_utils.get_model_inputs_from_batch(model, batch)
  assert not args
  return kwargs['tokens']
