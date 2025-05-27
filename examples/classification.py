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

r"""Example config for finetuning Gemma for a classification task.

* Input: A text to classify.
* Output: A classification label. The pre-trained Gemma model is trained to
  predict one world among 256.000. Here, we're finetuning to predict only 2
  tokens among the 256.000 available.

Train locally with:

```sh
python -m kauldron.main \
    --cfg=examples/classification.py \
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
  """Get the default hyperparameter configuration."""
  return kd.train.Trainer(
      seed=42,
      # Dataset
      train_ds=_make_dataset(training=True),
      # Model definition
      model=gm.nn.Gemma3_4B(
          tokens="batch.sentence",
          return_last_only=True,
      ),
      # Load the weights from the pretrained checkpoint
      init_transform=gm.ckpts.LoadCheckpoint(
          path=gm.ckpts.CheckpointPath.GEMMA3_4B_IT,
      ),
      # Training
      num_train_steps=10_000,
      train_losses={
          "xentropy": kd.losses.SoftmaxCrossEntropyWithIntLabels(
              logits="preds.logits",
              labels="batch.label",
          ),
      },
      optimizer=optax.adafactor(learning_rate=1e-4),
      checkpointer=kd.ckpts.Checkpointer(
          save_interval_steps=500,
      ),
      # Evaluation
      evals={
          "test": kd.evals.Evaluator(
              run=kd.evals.EveryNSteps(1000),
              ds=_make_dataset(training=False),
          ),
      },
  )


def _make_dataset(training: bool) -> kd.data.Pipeline:
  # Dict key names from the dataset
  _INPUT_FIELD = "sentence"  # pylint: disable=invalid-name
  _LABEL_FIELD = "label"  # pylint: disable=invalid-name

  tokenizer = gm.text.Gemma3Tokenizer()

  return kd.data.py.Tfds(
      name="glue/cola",
      split="train" if training else "validation",
      shuffle=True if training else False,
      num_epochs=None if training else 1,
      batch_size=8,
      transforms=[
          # Process the input text
          # TFDS datasets returns `bytes`, so convert them to `str`
          gm.data.DecodeBytes(key=_INPUT_FIELD),
          gm.data.FormatText(
              key=_INPUT_FIELD,
              template="""<start_of_turn>user
              Please classify whether the following sentence is grammaticaly correct, please answer only with Yes or No.
              Sentence: {text}<end_of_turn>
              <start_of_turn>model""",
          ),
          gm.data.Tokenize(
              key=_INPUT_FIELD,
              tokenizer=tokenizer,
              add_bos=True,
          ),
          gm.data.Pad(
              key=_INPUT_FIELD,
              max_length=128,
          ),
          # Process the label
          gm.data.MapInts(
              key=_LABEL_FIELD,
              # Rather than predicting the token 0 and 1, we are using the
              # token 1294 and 3553 which respectivelly correspond to "No" and
              # "Yes". We do this because those token already contain semantic
              # information, so even zero-shot prediction without any
              # finetuning has better than random performances.
              old_to_new={
                  0: 1294,  # Token -> "No"
                  1: 3553,  # Token -> "Yes"
              },
          ),
          kd.data.Rearrange(
              key=_LABEL_FIELD,
              pattern="... -> ... 1",  # For shape compatibility with the loss.
          ),
      ],
  )
