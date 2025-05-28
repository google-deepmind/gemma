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

from gemma import gm
from gemma.examples import seq2seq
from kauldron import konfig
import tensorflow_datasets as tfds

# Activate the fixture
use_hermetic_tokenizer = gm.testing.use_hermetic_tokenizer


def test_examples():
  cfg = seq2seq.get_config()
  with konfig.mock_modules():
    cfg.model = gm.testing.DummyGemma(
        tokens='batch.input',
    )
  cfg.train_ds.num_workers = 0  # Disable multi-processing in tests.
  cfg.workdir = '/tmp/gemma_test'

  trainer = konfig.resolve(cfg)

  # Resolve the training step, including the metrics, losses,...
  with tfds.testing.mock_data(num_examples=10):
    _ = trainer.context_specs
