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

r"""Example of Gemma finetuning for a prompt -> response task.

This is a fork of the next_token_prediction example, but with sharding.
The only difference is the `sharding=kd.sharding.ShardingStrategy()`

Train locally with:

```sh
python -m kauldron.main \
    --cfg=examples/sharding.py \
    --cfg.workdir=/tmp/kauldron_oss/workdir
```

"""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from gemma import gm
  from kauldron import kd
  import optax
# pylint: enable=g-import-not-at-top


def get_config():
  batch_size = 16
  max_length = 512

  return kd.train.Trainer(
      seed=42,
      # Dataset
      train_ds=_make_dataset(
          training=True,
          batch_size=batch_size,
          max_length=max_length,
      ),
      # Model definition
      model=gm.nn.Gemma2_2B(
          tokens="batch.input",
      ),
      sharding=kd.sharding.ShardingStrategy(
          params=kd.sharding.FSDPSharding(),
      ),
      # Load the weights from the pretrained checkpoint
      init_transform=gm.ckpts.LoadCheckpoint(
          path=gm.ckpts.CheckpointPath.GEMMA2_2B_IT,
      ),
      # Training
      num_train_steps=10_000,
      train_losses={
          "xentropy": kd.losses.SoftmaxCrossEntropyWithIntLabels(
              logits="preds.logits",
              labels="batch.target",
              mask="batch.loss_mask",
          ),
      },
      optimizer=optax.adafactor(learning_rate=1e-3),
      checkpointer=kd.ckpts.Checkpointer(
          save_interval_steps=500,
      ),
      # Evaluation
      evals={
          "test": kd.evals.Evaluator(
              run=kd.evals.EveryNSteps(1000),
              ds=_make_dataset(
                  training=False,
                  batch_size=batch_size,
                  max_length=max_length,
              ),
          ),
          # The sampler evaluator run inference on a few prompts from the
          # test set.
          "sampling": gm.evals.SamplerEvaluator(
              run=kd.evals.EveryNSteps(1000),
              # Sampling parameters
              tokenizer=gm.text.Gemma2Tokenizer(),
              max_new_tokens=50,
              # Which examples to use for sampling
              # The prompt and response indicates the fields to use within each
              # dataset example.
              prompt="prompt",
              response="response",
              num_examples=1,  # Only predict a single example
              ds=kd.data.py.Json(
                  shuffle=False,
                  num_epochs=1,
                  batch_size=None,
                  num_workers=0,
              ),
          ),
      },
  )


def _make_dataset(
    *,
    training: bool,
    batch_size: int,
    max_length: int,
):
  tokenizer = gm.text.Gemma2Tokenizer()

  split = "train" if training else "test"

  return kd.data.py.Json(
      shuffle=True if training else False,
      num_epochs=None if training else 1,
      batch_size=batch_size,
      num_workers=4,
      transforms=[
          gm.data.Tokenize(key="prompt", tokenizer=tokenizer, add_bos=True),
          gm.data.Tokenize(key="response", tokenizer=tokenizer, add_eos=True),
          # Create the model inputs/targets/loss_mask.
          gm.data.AddNextTokenPredictionFields(
              in_prompt="prompt",
              in_response="response",
              out_input="input",
              out_target="target",
              out_target_mask="loss_mask",
          ),
          # Only keep the fields we need.
          kd.data.Elements(keep=["input", "target", "loss_mask"]),
          # Pad the sequences to support batching.
          gm.data.Pad(
              key=["input", "target", "loss_mask"],
              max_length=max_length,
              # In this dataset, ~1% of examples are longer than 512 tokens.
              truncate=True,
          ),
          # For shape compatibility with the loss
          kd.data.Rearrange(
              key=["target", "loss_mask"], pattern="... -> ... 1"
          ),
      ],
  )
