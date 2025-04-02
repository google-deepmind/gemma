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

"""Example of using QLoRA (Quantized Low-Rank Adaptation) for fine-tuning Gemma 3.

QLoRA is a technique that combines quantized weights with Low-Rank Adapters
to enable parameter-efficient fine-tuning of large language models.
This example demonstrates QLoRA with Gemma 3 models (4B and 12B).

References:
  QLoRA: Efficient Finetuning of Quantized LLMs
  https://arxiv.org/abs/2305.14314
"""

import functools
import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import flax
from flax import jax_utils
from flax.training import checkpoints
from gemma.gm import ckpts as ckpt_lib
from gemma.gm import nn as gm_nn
from gemma.gm import text as gm_text
from gemma.gm.data import _tasks as tasks
from gemma.peft import _quantization_utils
from gemma.peft import _tree_utils
import jax
from jax import numpy as jnp
import optax

_CKPT_PATH = flags.DEFINE_string(
    "ckpt_path", None, "Path to checkpoint, relative to dir."
)
_OUTPUT_DIR = flags.DEFINE_string("output_dir", None, "Directory to save outputs.")
_MODEL_SIZE = flags.DEFINE_string(
    "model_size", "4b", "Size of model to load. Options: 4b, 12b."
)
_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate", 1e-4, "Learning rate for fine-tuning."
)
_RANK = flags.DEFINE_integer("rank", 8, "Rank of LoRA adapters.")
_EPOCHS = flags.DEFINE_integer("epochs", 3, "Number of training epochs.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
_SEQ_LEN = flags.DEFINE_integer("seq_len", 512, "Maximum sequence length.")
_SAVE_STEPS = flags.DEFINE_integer("save_steps", 100, "Save checkpoint every N steps.")


def create_model_and_tokenizer():
  """Initializes model, QLoRA wrapper, and tokenizer."""
  tokenizer = gm_text.Gemma3Tokenizer()

  # Initialize model based on size
  if _MODEL_SIZE.value == "4b":
    model_cls = gm_nn.Gemma3_4B
  elif _MODEL_SIZE.value == "12b":
    model_cls = gm_nn.Gemma3_12B
  else:
    raise ValueError(f"Invalid model size: {_MODEL_SIZE.value}")

  # Initialize the base model with text_only=True since we're not using vision capabilities
  base_model = model_cls(text_only=True)
  
  # Wrap with QLoRA
  model = gm_nn.QLoRA(
      model=base_model,
      rank=_RANK.value,
      quant_method=_quantization_utils.QuantizationMethod.INT4,
  )

  # Load the checkpoint
  variables = ckpt_lib.load_checkpoint(_CKPT_PATH.value, device_buffer=True)
  return model, variables, tokenizer


def create_train_state(model, variables, learning_rate=1e-4):
  """Creates a training state with optimizer."""
  # Split the original and QLoRA parameters
  params_original, params_lora = _tree_utils.split_params(variables["params"])

  # Create optimizer that only updates LoRA parameters
  optimizer = optax.adam(learning_rate=learning_rate)
  
  # Initialize TrainState with LoRA params for optimization
  class TrainState(flax.struct.PyTreeNode):
    tx: optax.GradientTransformation
    step: jnp.ndarray
    params_original: dict
    params_lora: dict
    opt_state: optax.OptState

  return TrainState(
      step=jnp.array(0),
      params_original=params_original,
      params_lora=params_lora,
      tx=optimizer,
      opt_state=optimizer.init(params_lora),
  )


def create_input_batch(tokenizer, texts, seq_len=512):
  """Tokenizes and creates a batch from input texts."""
  tokens = []
  for text in texts:
    token_ids = tokenizer.encode(text)
    # Pad or truncate to seq_len
    if len(token_ids) < seq_len:
      token_ids = token_ids + [tokenizer.pad_id] * (seq_len - len(token_ids))
    else:
      token_ids = token_ids[:seq_len]
    tokens.append(token_ids)
  
  return jnp.array(tokens)


def save_lora_checkpoint(state, step, output_dir):
  """Saves only the LoRA parameters."""
  checkpoint_path = os.path.join(output_dir, f"lora_checkpoint_{step}")
  checkpoints.save_checkpoint(
      checkpoint_path, state.params_lora, step=step, overwrite=True
  )
  return checkpoint_path


def load_dataset():
  """Loads a simple instruction-tuning dataset."""
  # For this example, use a simple synthetic dataset
  # In a real application, you would load an actual dataset
  examples = []
  for i in range(1000):
    examples.append({
        "instruction": f"Please summarize the following text #{i}",
        "input": f"This is a sample text #{i} that needs to be summarized.",
        "output": f"Sample text #{i} summary."
    })
  
  return examples


@functools.partial(jax.jit, static_argnums=(0,))
def train_step(model, state, batch_inputs, batch_targets, attention_mask=None):
  """Performs a single training step."""
  def loss_fn(params_lora):
    # Merge the original and LoRA parameters
    params = _tree_utils.merge_params(state.params_original, params_lora)
    
    # Forward pass
    logits = model.apply(
        {"params": params}, 
        batch_inputs, 
        attention_mask=attention_mask,
        enable_dropout=False,
    ).logits
    
    # Calculate loss (simple cross-entropy)
    targets_one_hot = jax.nn.one_hot(batch_targets, logits.shape[-1])
    loss = -jnp.sum(targets_one_hot * jax.nn.log_softmax(logits)) / batch_targets.size
    
    return loss
  
  # Compute gradients
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params_lora)
  
  # Apply gradients
  updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params_lora)
  new_params_lora = optax.apply_updates(state.params_lora, updates)
  
  # Update state
  new_state = state.replace(
      step=state.step + 1,
      params_lora=new_params_lora,
      opt_state=new_opt_state,
  )
  
  return new_state, loss


def prepare_instruction_data(tokenizer, example, seq_len):
  """Prepares instruction data for training."""
  # Format: instruction + input text + output
  formatted_input = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: "
  input_ids = tokenizer.encode(formatted_input)
  
  # Target: output text to be predicted
  target_ids = tokenizer.encode(example['output'])
  
  # Combined for loss calculation (shifted right for next-token prediction)
  combined = input_ids + target_ids
  if len(combined) > seq_len:
    combined = combined[:seq_len]
  
  # Inputs are tokens except the last one
  inputs = combined[:-1]
  # Add padding if needed
  if len(inputs) < seq_len - 1:
    inputs = inputs + [tokenizer.pad_id] * (seq_len - 1 - len(inputs))
  
  # Targets are tokens shifted by one (predict next token)
  targets = combined[1:]
  if len(targets) < seq_len - 1:
    targets = targets + [tokenizer.pad_id] * (seq_len - 1 - len(targets))
    
  return jnp.array(inputs), jnp.array(targets)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  
  # Create output directory
  if not os.path.exists(_OUTPUT_DIR.value):
    os.makedirs(_OUTPUT_DIR.value)
  
  # Initialize model and tokenizer
  model, variables, tokenizer = create_model_and_tokenizer()
  
  # Create training state
  state = create_train_state(model, variables, learning_rate=_LEARNING_RATE.value)
  
  # Load dataset
  dataset = load_dataset()
  
  # Training loop
  steps_per_epoch = len(dataset) // _BATCH_SIZE.value
  total_steps = steps_per_epoch * _EPOCHS.value
  
  logging.info("Starting QLoRA fine-tuning for Gemma 3")
  logging.info(f"Model: Gemma 3 {_MODEL_SIZE.value}, Rank: {_RANK.value}")
  logging.info(f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}")
  
  for step in range(total_steps):
    # Get batch
    batch_idx = step % steps_per_epoch
    batch_data = dataset[batch_idx * _BATCH_SIZE.value:(batch_idx + 1) * _BATCH_SIZE.value]
    
    # Process data
    batch_inputs = []
    batch_targets = []
    for example in batch_data:
      inputs, targets = prepare_instruction_data(tokenizer, example, _SEQ_LEN.value)
      batch_inputs.append(inputs)
      batch_targets.append(targets)
    
    # Stack batches
    batch_inputs = jnp.stack(batch_inputs)
    batch_targets = jnp.stack(batch_targets)
    
    # Create attention mask (1 for tokens, 0 for padding)
    attention_mask = (batch_inputs != tokenizer.pad_id).astype(jnp.int32)
    
    # Training step
    state, loss = train_step(model, state, batch_inputs, batch_targets, attention_mask)
    
    # Log progress
    if step % 10 == 0:
      logging.info(f"Step {step}/{total_steps}, Loss: {loss}")
    
    # Save checkpoint
    if step % _SAVE_STEPS.value == 0 or step == total_steps - 1:
      save_lora_checkpoint(state, step, _OUTPUT_DIR.value)
  
  # Save final QLoRA parameters
  final_checkpoint_path = save_lora_checkpoint(state, total_steps, _OUTPUT_DIR.value)
  logging.info(f"Training completed. Final checkpoint saved at: {final_checkpoint_path}")


if __name__ == "__main__":
  app.run(main)
