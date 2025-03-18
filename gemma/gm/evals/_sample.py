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

from collections.abc import Iterable, Iterator, Mapping
import dataclasses
import functools
import itertools
from typing import Any, Optional

from etils import epy
from flax import linen as nn
from gemma.gm.data import _tasks
from gemma.gm.text import _sampler
import jax
from kauldron import kd
from kauldron.utils import config_util
from kauldron.utils import immutabledict


def batched_iterable(
    iterable: Iterable[Any], batch_size: int
) -> Iterator[list[Any]]:
  """Yields lists of elements from an iterable in batches."""
  it = iter(iterable)
  while batch := list(itertools.islice(it, batch_size)):
    yield batch


class SamplerEvaluator(kd.evals.EvaluatorBase):
  """Sampling evaluator that supports batched sampling.

  The evaluator expects as dataset containing a `Seq2SeqTask` transform.

  Attributes:
    max_new_tokens: Maximum number of new tokens to generate for each sample. In
      total, the model will process `input_length + max_new_tokens`.
    num_examples: How many examples to sample in total. If None, all examples
      from the dataset will be used.
    batch_size: The number of examples to process in parallel for each sampling
      operation.
    ds: Dataset to evaluate on. Note that the dataset must be unbatched and
      contain raw `str` fields.
    model: The model to use.
    losses: Losses to compute. Losses and metrics can access the prediction text
      through the key: `preds.text`.
    metrics: Metrics to compute. Losses and metrics can access the prediction
      text through the key: `preds.text`.
    summaries: Optional summaries to write.
  """

  # Sampler parameters
  max_new_tokens: int

  # Dataset parameters
  num_examples: Optional[int] = 1
  batch_size: int = 1
  ds: kd.data.Pipeline = config_util.ROOT_CFG_REF.eval_ds

  model: nn.Module = config_util.ROOT_CFG_REF.model

  # Losses, metrics, summaries
  losses: Mapping[str, kd.losses.Loss] = dataclasses.field(default_factory=dict)
  metrics: Mapping[str, kd.metrics.Metric] = dataclasses.field(
      default_factory=dict
  )
  summaries: Mapping[str, kd.summaries.Summary] = dataclasses.field(
      default_factory=dict
  )

  def __post_init__(self):
    super().__post_init__()
    immutabledict.freeze_dict_attrs(self, ['losses', 'metrics', 'summaries'])

  def evaluate(self, state: kd.train.TrainState, step: int) -> Any:
    """Run this evaluator then write and optionally return the results."""
    self._assert_root_cfg_resolved()

    prompts = []
    samples = []

    # Evaluators are run within the `jax.transfer_guard('disallow')` for the
    # train loop, so re-enable it here.
    with jax.transfer_guard('allow'):
      sampler = _sampler.Sampler(
          model=self.model,
          params=state.params,
          tokenizer=self._task.tokenizer,
      )
      for examples_batch in batched_iterable(self.examples, self.batch_size):
        prompts_batch = [
            kd.kontext.get_by_path(ex, self._task.out_input)
            for ex in examples_batch
        ]

        # Extract the images input from the dataset.
        images_batch = None
        if self.model.images is not None:
          images_batch = [
              kd.kontext.get_by_path({'batch': ex}, self.model.images)
              for ex in examples_batch
          ]

        # TODO(epot): Which sharding for the images ?
        images_batch = kd.sharding.device_put(
            images_batch, kd.sharding.REPLICATED
        )

        samples = sampler.sample(
            prompts_batch,
            images=images_batch,
            max_new_tokens=self.max_new_tokens,
        )
        prompts.extend(prompts_batch)
        samples.extend(samples)

    # ======= Write outputs =======

    aux = kd.train.Auxiliaries(
        losses=self.losses,
        metrics=self.metrics,
        summaries=self.summaries,
    )
    aux_state = kd.train.AuxiliariesState()

    # TODO(epot): Would be nice to also write the samples to some text file,
    # Like `self.eval_path(step) / f'{self.name}_preds.txt'` or
    # `self.writer.path_for_step(step) ?`
    # TODO(epot): If that's the case, also create a XManager artifact ?
    texts = {}
    for i, (prompt, ex, sample) in enumerate(
        zip(prompts, self.examples, samples)
    ):
      # TODO(epot): Should instead uses summaries.
      # Text summaries
      texts[f'prompt_{i}'] = prompt
      texts[f'sample_{i}'] = sample
      if self._task.out_target:
        texts[f'gt_{i}'] = kd.kontext.get_by_path(ex, self._task.out_target)

      # Images & other summaries
      with jax.transfer_guard('allow'):
        # TODO(epot): Should simplify the abstractions (i.e. merge
        # `update_context` & `get_aux_state`).
        ctx = kd.train.Context(
            step=step,
            batch=ex,
            preds={'text': sample},
        )
        ctx = aux.update_context(ctx)
        new_state = ctx.get_aux_state(
            return_losses=True,
            return_metrics=True,
            return_summaries=True,
        )
        aux_state |= new_state

    # Write the data to the writer.
    self.writer.write_texts(step, texts)
    self.writer.write_step_metrics(
        step=step,
        aux=aux_state,
        schedules={},
        log_summaries=True,
    )

  @functools.cached_property
  def examples(self) -> list[Any]:
    """Extract the prompts from the dataset."""
    return list(self.ds.take(self.num_examples))

  # TODO(epot): More flexible way to connect the dataset (e.g. eval-only
  # datasets). Supports custom `Seq2Seq` transforms,...
  # Could be done by adding some `__sampling_info__` protocol to the
  # `Seq2SeqTask`
  @functools.cached_property
  def _task(self) -> _tasks.Seq2SeqTask:
    """Extract the `Seq2SeqTask` from the dataset."""
    if not isinstance(self.ds, kd.data.py.PyGrainPipeline):
      raise ValueError(
          'SamplerEvaluator only supports `PyGrainPipeline` datasets.'
      )
    # TODO(epot): Supports mixture or more complex datasets.
    for t in self.ds.transforms:
      if isinstance(t, _tasks.Seq2SeqTask):
        return t
    raise ValueError(
        'Could not find a `Seq2SeqTask` transform in the dataset. This is'
        f' required by SamplerEvaluator. Dataset: {epy.pretty_repr(self.ds)}'
    )

  @functools.cached_property
  def __dashboards__(self) -> kd.kdash.DashboardsBase:
    """Returns collection keys used by flat board."""
    # This evaluator do not report any scalar metrics.
    return kd.kdash.NoopDashboard()
