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

r"""Example of Gemma finetuning using QLoRA.

This example is based on the `lora.py` example. See that file
and the `seq2seq.py` example for more details.

QLoRA (Quantized Low-Rank Adaptation) combines parameter-efficient
fine-tuning with quantized weights to significantly reduce
memory requirements while maintaining performance.

The changes to use QLoRA are:

* `model`: Use `gm.nn.QLoRA()` wrapper to add quantized layers with
  `LoRA` adapters to the model.
* `init_transform`: Use `gm.ckpts.SkipLoRA()` wrapper to only restore the
  non-LoRA weights.
* `optimizer`: Use `kd.optim.partial_updates` wrapper to only train the LoRA
  weights.

Train locally with:

```sh
python -m kauldron.main \
    --cfg=examples/qlora.py \
    --cfg.workdir=/tmp/kauldron_oss/workdir
```

"""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from gemma import gm
  from gemma.peft import _quantization_utils
  from kauldron import kd
  import optax
# pylint: enable=g-import-not-at-top


def get_config():
  batch_size = 16
  max_length = 512
  rank = 8  # Rank for LoRA adapters

  return kd.train.Trainer(
      seed=42,
      # Dataset
      train_ds=_make_dataset(
          training=True,
          batch_size=batch_size,
          max_length=max_length,
      ),
      # Model definition with QLoRA
      model=gm.nn.QLoRA(
          rank=rank,
          quant_method=_quantization_utils.QuantizationMethod.INT4,
          model=gm.nn.Gemma3_4B(
              tokens="batch.input",
              text_only=True,  # QLoRA currently doesn't support multimodal
          ),
      ),
      # Load the weights from the pretrained checkpoint
      # Use `SkipLoRA` as the original checkpoint does not contain the LoRA
      # weights.
      init_transform=gm.ckpts.SkipLoRA(
          wrapped=gm.ckpts.LoadCheckpoint(
              path=gm.ckpts.CheckpointPath.GEMMA3_4B_IT,
          )
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
      # Use a lower learning rate when fine-tuning with QLoRA
      optimizer=kd.optim.partial_updates(
          optax.adafactor(learning_rate=1e-4),
          # We only optimize the LoRA weights. The rest of the model is frozen.
          mask=kd.optim.select("lora"),
      ),
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
              max_new_tokens=150,  # Sampling parameters
              num_batches=1,  # Only predict a single example (batch_size=None)
              ds=_make_dataset(training=False, sampling=True),
          ),
      },
  )


def _make_dataset(
    *,
    training: bool,
    sampling: bool = False,
    batch_size: int | None = None,
    max_length: int | None = None,
):
  tokenizer = gm.text.Gemma3Tokenizer()

  return kd.data.py.Tfds(
      name="mtnt/en-fr",
      split="train" if training else "test",
      shuffle=True if training else False,
      num_epochs=None if training else 1,
      batch_size=None if sampling else batch_size,
      num_workers=4,
      transforms=[
          # Create the model inputs/targets/loss_mask.
          gm.data.Seq2SeqTask(
              # Select which field from the dataset to use.
              # https://www.tensorflow.org/datasets/catalog/mtnt
              in_prompt="src",
              in_response="dst",
              # Output batch is {"input": ..., "target": ..., "loss_mask": ...}
              out_input="input",
              out_target="target",
              out_target_mask="loss_mask",
              tokenizer=tokenizer,
              # Padding parameters
              max_length=None if sampling else max_length,
              # In this dataset, ~1% of examples are longer than 512 tokens.
              truncate=True,
              sampling=sampling,
          ),
      ],
  )