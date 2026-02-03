# Copyright 2026 DeepMind Technologies Limited.
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

r"""NPO Example.

NPO [https://arxiv.org/pdf/2404.05868] is an unlearning algorithm that is
a modification of DPO. Like DPO, it uses two models to generate the response.
But NPO only focuses on reducing the answer likelihood compared to the reference model.

L_NPO = − E[log σ(− β log{π_θ(y|x)/π_ref (y|x)})


Train locally with:

```sh
python -m kauldron.main \
    --cfg=examples/npo.py \
    --cfg.workdir=/tmp/kauldron_oss/workdir
```

"""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from gemma import gm
  from kauldron import kd
  import optax
  import tensorflow_datasets as tfds
# pylint: enable=g-import-not-at-top


def get_config():
  """Get the default hyperparameter configuration."""

  return kd.train.Trainer(
      seed=42,
      # Dataset
      train_ds=_make_dataset(
          dataset_name="locuslab__tofu/forget10",
          data_dir=tfds.HF_PLACER_DIR,
      ),
      # Model definition
      model=gm.nn.AnchoredPolicy(
          policy=gm.nn.Gemma3_12B(tokens="batch.input", text_only=True),
      ),
      # Load the weights from the pretrained checkpoint
      init_transform=gm.ckpts.AnchoredPolicyLoader(
          policy=gm.ckpts.LoadCheckpoint(
              path=gm.ckpts.CheckpointPath.GEMMA3_12B_IT,
          ),
      ),
      # Training
      num_train_steps=250,
      train_losses={
          "npo": gm.losses.NpoLoss(
              tokens="batch.target",
              sequence_mask="batch.mask",
              policy_logits="preds.policy.logits",
              anchor_logits="preds.anchor.logits",
          ),
      },
      optimizer=optax.adafactor(learning_rate=1e-4),
      checkpointer=kd.ckpts.Checkpointer(
          save_interval_steps=50,
      ),
      # Evaluation
      evals={
          # "test": kd.evals.Evaluator(
          #     run=kd.evals.EveryNSteps(1000),
          #     ds=_make_dataset(
          #         training=False, dataset_name=dataset_name, data_dir=data_dir
          #     ),
          # ),
      },
  )


def _make_dataset(
    dataset_name: str, data_dir: str
):
  """Make a dataset for training or evaluation.

  Args:
    dataset_name: The name of the dataset to load.
    data_dir: The directory where the dataset is located.

  Returns:
    A `kd.data.Pipeline` for the dataset.
  """
  max_length = 256
  batch_size = 16

  tokenizer = gm.text.Gemma3Tokenizer()

  return kd.data.py.Tfds(
      name=dataset_name,
      data_dir=data_dir,
      split="all",
      shuffle=True,
      batch_size=batch_size,
      transforms=[
          # Drop the 'split' column
          kd.data.py.Elements(keep=["question", "answer"]),
          # Create the model inputs/targets/mask.
          gm.data.Seq2SeqTask(
              in_prompt="question",
              in_response="answer",
              # Output batch is {'input': ..., 'target': ..., 'mask': ...}
              out_input="input",
              out_target="target",
              out_target_mask="mask",
              tokenizer=tokenizer,
              # Padding parameters
              max_length=max_length,
              truncate=True,
          ),
      ],
  )
