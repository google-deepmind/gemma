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

r"""DPO Example.

DPO works by running two answers (one prefered and one rejected) into both
the reference model and the model to finetune. Then the DPO loss is used to
increase the likelihood of generating the preferred answer.

Implementation wise, this is done by:

* Wrapping the model inside a `gm.nn.AnchoredPolicy` (which runs both the
  model and the reference frozen model)
* Using the `gm.ckpts.AnchoredPolicyLoader` to restore the weights, so the
  weights are correctly mapped to inside `gm.nn.AnchoredPolicy`.


Train locally with:

```sh
python -m kauldron.main \
    --cfg=examples/dpo.py \
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
      model=gm.nn.AnchoredPolicy(
          policy=gm.nn.Gemma3_4B(tokens="batch.tokens", text_only=True),
      ),
      # Load the weights from the pretrained checkpoint
      init_transform=gm.ckpts.AnchoredPolicyLoader(
          policy=gm.ckpts.LoadCheckpoint(
              path=gm.ckpts.CheckpointPath.GEMMA3_4B_IT,
          ),
      ),
      # Training
      num_train_steps=10_000,
      train_losses={
          "dpo": gm.losses.DpoLoss(
              tokens="batch.targets",
              sequence_mask="batch.mask",
              policy_logits="preds.policy.logits",
              anchor_logits="preds.anchor.logits",
          ),
      },
      optimizer=optax.adafactor(learning_rate=1e-4),
      checkpointer=kd.ckpts.Checkpointer(
          save_interval_steps=500,
      ),
      # Evaluation
      evals={
          # "test": kd.evals.Evaluator(
          #     run=kd.evals.EveryNSteps(1000),
          #     ds=_make_dataset(training=False),
          # ),
      },
  )


def _make_dataset(training: bool) -> kd.data.Pipeline:
  # TODO(epot): !!!!
  max_length = 512
  batch_size = 16

  tokenizer = gm.text.Gemma3Tokenizer()

  return kd.data.py.HuggingFace(
      path="argilla/distilabel-math-preference-dpo",
      split="train",
      shuffle=True if training else False,
      num_epochs=None if training else 1,
      batch_size=batch_size,
      transforms=[
          # Only keep the fields we need.
          kd.data.Elements(
              keep=["instruction", "chosen_response", "rejected_response"]
          ),
          # Create the model inputs and loss mask.
          gm.data.ContrastiveTask(
              in_prompt="instruction",
              in_chosen="chosen_response",
              in_rejected="rejected_response",
              out_tokens="tokens",
              out_targets="targets",
              out_mask="mask",
              tokenizer=tokenizer,
              # Padding parameters
              max_length=max_length,
              # TODO(epot): Run stats (how many examples are we dropping?)
              truncate=True,
          ),
      ],
  )
