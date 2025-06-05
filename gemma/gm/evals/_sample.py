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

"""Sampling evaluator."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import functools
from typing import Any, Optional

from etils import epy
from flax import linen as nn
from gemma.gm.data import _tasks
from gemma.gm.text import _sampler
import jax
from kauldron import kd
from kauldron.utils import config_util
from kauldron.utils import immutabledict
from kauldron.utils import utils


class SamplerEvaluator(kd.evals.EvaluatorBase):
  """Sampling evaluator.

  The evaluator expects as dataset containing a `Seq2SeqTask` transform.

  Attributes:
    cache_length: Cache length to use. This is the maximum number of tokens the
      conversation can have (prompts, answers, images for all turns). Setting
      this to a fixed value avoids re-compilation between turns.
    max_new_tokens: Maximum number of new tokens to generate. In total, the
      model will process `input_length + max_new_tokens`.
    pad_length: Pad length for the input. This is useful to ensure the prompt is
      always the same length during sampling, which can be helful to avoid
      re-compilation.
    num_batches: Number of batches. If `None`, sample the entire dataset.
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
  cache_length: int = 4096
  max_new_tokens: int
  pad_length: int | None = None

  # Dataset parameters
  num_batches: Optional[int] = None
  cache: bool = False
  ds: kd.data.Pipeline = config_util.ROOT_CFG_REF.eval_ds

  model: nn.Module = config_util.ROOT_CFG_REF.model

  # Losses, metrics, summaries
  losses: Mapping[str, kd.losses.Loss] = dataclasses.field(default_factory=dict)
  metrics: Mapping[str, kd.metrics.Metric] = dataclasses.field(
      default_factory=dict
  )
  summaries: Mapping[str, kd.metrics.Metric] = dataclasses.field(
      default_factory=dict
  )

  def __post_init__(self):
    super().__post_init__()
    immutabledict.freeze_dict_attrs(self, ['losses', 'metrics', 'summaries'])

  def evaluate(self, state: kd.train.TrainState, step: int) -> Any:
    """Run this evaluator then write and optionally return the results."""
    self._assert_root_cfg_resolved()

    sampler = self._get_sampler(state)

    # TODO(epot): Better sharding, a few options:
    #  1. Allow to customize the sharding to process examples
    #  2. Or auto-detect the sharding to set to `FIRST_DIM` when possible.
    #  3. Re-use sharding from `trainer.sharding.ds`
    if self.ds.batch_size is None:
      sharding = kd.sharding.REPLICATED
    else:
      sharding = kd.sharding.FIRST_DIM

    # Default text summaries
    summaries = {
        'prompt': kd.summaries.ShowTexts(texts=f'batch.{self._task.out_input}'),
        'answer': kd.summaries.ShowTexts(texts='preds.text'),
    }
    if self._task.out_target:
      summaries['gt'] = kd.summaries.ShowTexts(
          texts=f'batch.{self._task.out_target}'
      )

    # Accumulate metrics.
    aux = kd.train.Auxiliaries(
        losses=self.losses,
        metrics=self.metrics,
        summaries=summaries | self.summaries,
    )
    aux_state = kd.train.AuxiliariesState()

    # TODO(epot): Would be nice to also write the samples to some text file,
    # Like `self.eval_path(step) / f'{self.name}_preds.txt'` or
    # `self.writer.path_for_step(step) ?`
    # TODO(epot): If that's the case, also create a XManager artifact ?

    # Evaluators are run within the `jax.transfer_guard('disallow')` for the
    # train loop, so re-enable it here.
    with jax.transfer_guard('allow'):

      for _, ex in utils.enum_iter(self.ds_iter, desc=self.name):
        prompts, images = self._get_prompt_and_image_from_batch(ex)

        images = kd.sharding.device_put(images, sharding)
        out_text = sampler.sample(
            prompts,
            images=images,
            max_new_tokens=self.max_new_tokens,
            sharding=sharding,
        )

        # TODO(epot): Should simplify the abstractions (i.e. merge
        # `update_context` & `get_aux_state`).
        ctx = kd.train.Context(
            step=step,
            batch=ex,
            preds={'text': out_text},
        )
        ctx = aux.update_context(ctx)
        new_state = ctx.get_aux_state(
            return_losses=True,
            return_metrics=True,
            return_summaries=True,
        )
        aux_state |= new_state

    # ======= Write outputs =======

    # Write the data to the writer.
    self.writer.write_step_metrics(
        step=step,
        aux=aux_state,
        schedules={},
        log_summaries=True,
    )

  @functools.cached_property
  def ds_iter(self) -> kd.data.IterableDataset:
    """Iterate over the examples."""
    ds_iter = self.ds
    if self.num_batches is not None:
      ds_iter = ds_iter.take(self.num_batches)
    if self.cache:
      if self.num_batches is None:
        raise ValueError('Can only cache if num_batches is set.')
      ds_iter = ds_iter.cache()
    return ds_iter

  def _get_sampler(self, state: kd.train.TrainState):
    """Returns the sampler to use."""
    return _sampler.Sampler(
        model=self.model,
        params=state.params,
        tokenizer=self._task.tokenizer,
        cache_length=self.cache_length,
        pad_length=self.pad_length,
    )

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

  def _get_prompt_and_image_from_batch(self, ex: Any) -> tuple[str, Any]:
    """Returns the prompt and image from the example."""
    # self._task.out_input == self.model.tokens
    prompt = kd.kontext.get_by_path(ex, self._task.out_input)

    # Extract the images input from the dataset.
    if self.model.images is not None:
      images = kd.kontext.get_by_path({'batch': ex}, self.model.images)
    else:
      images = None
    return prompt, images

  @functools.cached_property
  def __dashboards__(self) -> kd.kdash.DashboardsBase:
    """Returns collection keys used by flat board."""
    # This evaluator do not report any scalar metrics.
    return kd.kdash.NoopDashboard()
