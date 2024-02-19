# This is a Colab notebook. Consider opening in http://colab.google/ or as
# a notebook in JupyterLab.
# %% [markdown]
# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---
# %% [markdown]
# # Fine-tuning the 2B Gemma model with flax
#
# In this tutorial you will learn how to fine-tune the 2B Gemma model for a simple translation task. To run this colab, you will need to use a TPU v4 runtime.
# %% [markdown]
# ## Setup
# %% Installation
# ## Setup
#
# Please follow installation instructions at https://github.com/google-deepmind/gemma/blob/main/README.md.

# %% Download the checkpoints
# Download a Flax checkpoint and a tokenizer from http://kaggle.com/models/google/gemma.
# Set the local paths below.
ckpt_path = ''
vocab_path = ''
# %% Python imports

import enum
import re
import string

# We import JAX and some related packages.
import chex
import jax
import jax.numpy as jnp
import optax

# We will use tensorflow to handle the dataset
import tensorflow as tf
import tensorflow_datasets as tfds

# Finally, we import Gemma.
from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib
import sentencepiece as spm
# %% [markdown]
# ## Step 1: prepare the dataset
#
# ### The MTNT dataset
#
# In this tutorial, we will use the MTNT dataset, from the paper [MTNT: A Testbed for Machine Translation of Noisy Text](https://arxiv.org/abs/1809.00388). This dataset is directly available in the [TensorFlow dataset catalog](https://www.tensorflow.org/datasets/catalog/mtnt).
#
# More precisely we will focus on the English to French translation.
#
# But let's have a look at the data themselves.
# %%
ds = tfds.load("mtnt/en-fr", split="train")
ds = ds.take(2)
ds = ds.as_numpy_iterator()
for idx, example in enumerate(ds):
  print(f'Example {idx}:')
  for key, val in example.items():
    print(f'{key}: {val}')
  print()
# %% [markdown]
# Each sample in the dataset contains two entries:
# - 'src': The original English sentence.
# - 'dst': The corresponding French translation.
# %% [markdown]
# ### Tokenizer
#
# Let's start by loading our vocabulary base tokenizer, which we'll construct using the [SentencePiece](https://github.com/google/sentencepiece) library.
# %%
vocab = spm.SentencePieceProcessor()
vocab.Load(vocab_path)
# %% [markdown]
# Let's customize `SentencePieceProcessor` for our English-to-French translation task. Since we're fine-tuning the English-only Gemma 2B model, we need a few adjustments:
#
# - **Input Prefix**: Adding a common prefix to each input signals the translation task. For example we could go with a prompt like `Translate this into French: [INPUT_SENTENCE]`.
#
# - **Translation Start suffix**: We add a suffix at the end of each prompt tells the model exactly when to begin the translation process. A new line should do the job.
#
# - **LM Tokens**: Gemma models expect a *beginning of sequence* token at the beginning of each sequence. Similarly, we need to add an *end of sequence* token at the end of each training example.
# %%
class GemmaTokenizer:
  """Custom wrapper around a SentencePieceProcessor for tensorflow."""

  def __init__(self,
               spm_processor: spm.SentencePieceProcessor):
    self._spm_processor = spm_processor

  @property
  def pad_id(self) -> int:
    """Fast access to the pad id."""
    return self._spm_processor.pad_id()

  def tokenize(self,
               example: str | bytes,
               prefix: str = '',
               suffix: str = '',
               add_eos: bool = True) -> jax.Array:
    """
    Tokenization function.

    Args:
      example: input string to tokenize.
      prefix:  prefix to add to the input string.
      suffix:  suffix to add to the input string.
      add_eos: if True, add an end of sentence token at the end of the output
               sequence.
    Returns:
      Tokens corresponding to the input string.
    """
    int_list = [self._spm_processor.bos_id()]
    int_list.extend(self._spm_processor.EncodeAsIds(prefix + example + suffix))
    if add_eos:
      int_list.append(self._spm_processor.eos_id())

    return jnp.array(int_list, dtype=jnp.int32)

  def tokenize_tf_op(self,
                     str_tensor: tf.Tensor,
                     prefix: str = '',
                     suffix: str = '',
                     add_eos: bool = True) -> tf.Tensor:
    """Tensforflow operator for the tokenize function."""
    encoded = tf.numpy_function(
        self.tokenize,
        [str_tensor, prefix, suffix, add_eos],
        tf.int32)
    encoded.set_shape([None])
    return encoded

  def to_string(self, tokens: jax.Array) -> str:
    """Convert an array of tokens to a string."""
    return self._spm_processor.EncodeIds(tokens.tolist())
# %% [markdown]
# Now let's try our custom tokenizer on the MTNT dataset
# %%
tokenizer = GemmaTokenizer(vocab)

def tokenize_source(tokenizer, example: tf.Tensor):
  return tokenizer.tokenize_tf_op(example,
                                  prefix='Translate this into French:\n',
                                  suffix='\n',
                                  add_eos=False)
def tokenize_destination(tokenizer, example: tf.Tensor):
  return tokenizer.tokenize_tf_op(example,
                                  add_eos=True)

ds = tfds.load("mtnt/en-fr",split="train")
ds = ds.take(2)
ds = ds.map(lambda x: {'src': tokenize_source(tokenizer, x['src']),
                       'dst': tokenize_destination(tokenizer, x['dst'])})
ds = ds.as_numpy_iterator()
for idx, example in enumerate(ds):
  print(f'Example {idx}:')
  for key, val in example.items():
    print(f'{key}: {val}')
  print()
# %% [markdown]
# ### Data loader
#
# We can now wrap everything a build our data loader.
# %%
@chex.dataclass(frozen=True)
class TrainingInput:
  # Input tokens given to the model
  input_tokens: jax.Array

  # A mask that determines which tokens contribute to the target loss
  # calculation.
  target_mask: jax.Array

class DatasetSplit(enum.Enum):
  TRAIN = 'train'
  VALIDATION = 'valid'


class MTNTDatasetBuilder:
  """Data loader for the MTNT dataset."""

  N_ITEMS = {DatasetSplit.TRAIN: 35_692,
             DatasetSplit.VALIDATION: 811}

  BUFFER_SIZE_SHUFFLE = 10_000
  TRANSLATION_PREFIX = 'Translate this into French:\n'
  TRANSLATION_SUFFIX = '\n'

  def __init__(self,
               tokenizer : GemmaTokenizer,
               max_seq_len: int):
    """Constructor.

    Args:
      tokenizer: Gemma tokenizer to use.
      max_seq_len: size of each sequence in a given batch.
    """
    self._tokenizer = tokenizer
    self._base_data = {
        DatasetSplit.TRAIN: tfds.load("mtnt/en-fr",split="train"),
        DatasetSplit.VALIDATION: tfds.load("mtnt/en-fr",split="valid"),
    }
    self._max_seq_len = max_seq_len

  def _tokenize_source(self, example: tf.Tensor):
    """Tokenization function for the source."""
    return self._tokenizer.tokenize_tf_op(example,
                                          prefix=self.TRANSLATION_PREFIX,
                                          suffix=self.TRANSLATION_SUFFIX,
                                          add_eos=False)

  def _tokenize_destination(self, example: tf.Tensor):
    """Tokenization function for the French translation."""
    return self._tokenizer.tokenize_tf_op(example,
                                          add_eos=True)

  def _pad_up_to_max_len(self,
                         input_tensor: tf.Tensor,
                         pad_value: int | bool,
                         ) -> tf.Tensor:
    """Pad the given tensor up to sequence length of a batch."""
    seq_len = tf.shape(input_tensor)[0]
    to_pad = tf.maximum(self._max_seq_len - seq_len, 0)
    return tf.pad(input_tensor,
                  [[0, to_pad]],
                  mode='CONSTANT',
                  constant_values=pad_value,
                  )

  def _to_training_input(self,
                         src_tokens: jax.Array,
                         dst_tokens: jax.Array,
                         ) -> TrainingInput:
    """Build a training input from a tuple of source and destination tokens."""

    # The input sequence fed to the model is simply the concatenation of the
    # source and the destination.
    tokens = tf.concat([src_tokens, dst_tokens], axis=0)

    # We want to prevent the model from updating based on the source (input)
    # tokens. To achieve this, we add a target mask to each input.
    q_mask = tf.zeros_like(src_tokens, dtype=tf.bool)
    a_mask = tf.ones_like(dst_tokens, dtype=tf.bool)
    mask = tf.concat([q_mask, a_mask], axis=0)

    # If the output tokens sequence is smaller than the target sequence size,
    # then we pad it with pad tokens.
    tokens = self._pad_up_to_max_len(tokens, self._tokenizer.pad_id)

    # We don't want to perform the backward on the pad tokens.
    mask = self._pad_up_to_max_len(mask, False)

    return TrainingInput(input_tokens=tokens, target_mask=mask)


  def get_train_dataset(self, batch_size: int, num_epochs: int):
    """Build the training dataset."""

    # Tokenize each sample
    ds = self._base_data[DatasetSplit.TRAIN].map(lambda x : (self._tokenize_source(x['src']),
                                                             self._tokenize_destination(x['dst'])))

    # Convert them to training inputs
    ds = ds.map(lambda x, y: self._to_training_input(x, y))

    # Remove the samples which are too long
    ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)

    # Shuffle the dataset
    ds = ds.shuffle(buffer_size=self.BUFFER_SIZE_SHUFFLE)

    # Repeat if necessary
    ds = ds.repeat(num_epochs)

    # Build batches
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

  def get_validation_dataset(self, batch_size: int):
    """Build the validation dataset."""

    # Same as the training dataset, but no shuffling and no repetition
    ds = self._base_data[DatasetSplit.VALIDATION].map(lambda x : (self._tokenize_source(x['src']),
                                                                  self._tokenize_destination(x['dst'])))
    ds = ds.map(lambda x, y: self._to_training_input(x, y))
    ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds
# %% [markdown]
# Let's give it a try.
# %%
tokenizer = GemmaTokenizer(vocab)
dataset_builder = MTNTDatasetBuilder(tokenizer, max_seq_len=20)
ds = dataset_builder.get_train_dataset(3, 1)
ds = ds.take(2)
ds = ds.as_numpy_iterator()
for idx, example in enumerate(ds):
  print(f'Example {idx}:')
  for key, val in example.items():
    print(f'{key}: {val}')
  print()
# %% [markdown]
# ## Fine tuning the Gemma model
#
# ### Getting started
#
# First let's load the model
# %%
# Load parameters

# TODO: change once the downloading url is known
params = params_lib.load_and_format_params(ckpt_path)

# We use the `transformer_lib.TransformerConfig.from_params` function to
# automatically load the correct configuration from a checkpoint. Note that the
# vocabulary size is smaller than the number of input embeddings due to unused
# tokens in this release.
config_2b = transformer_lib.TransformerConfig.from_params(
    params,
    cache_size=30  # Number of time steps in the transformer's cache
)
model_2b = transformer_lib.Transformer(config=config_2b)
# %% [markdown]
# Can our model translate French ? Well let's try it out !
# %%
sampler_old = sampler_lib.Sampler(
    transformer=model_2b,
    vocab=vocab,
    params=params['transformer'],
)
# %%
print(sampler_old(
    ["Translate this into French:\nHello, my name is Morgane.\n"],
    # number of steps performed when generating
    total_generation_steps=30,
  ).text)
# %% [markdown]
# As expected, it didn't work. Let's see if we can get better results by fine-tuning.
#
# Before moving further, don't forget to clear the memory if necessary.
# %%
del sampler_old
# %% [markdown]
# ### Model forward and loss function
#
# Gemma `Transformer` class inherits from [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html). It offers two essential methods:
#
# - `init`: Initializes the model's parameters.
#
# - `apply`: Executes the model's `__call__` function using a given set of parameters.
#
# Since are working with pre-trained weights, we won't use the `init` function.
#
# We define a `forward_and_loss_fn` as follows:
# %%
def forward_and_loss_fn(params,
                        *,
                        model: transformer_lib.Transformer,
                        input_tokens: jax.Array,            # Shape [B, L]
                        input_mask: jax.Array,              # Shape [B, L]
                        positions: jax.Array,               # Shape [B, L]
                        attention_mask: jax.Array,          # [B, L, L]
                        ) -> jax.Array:
  """Foward pass and loss function.

  Args:
    params: model's input parameters.
    model: gemma transformer model to call.
    input_tokens: input tokens sequence, shape [B, L].
    input_mask: tokens to ignore when computing the loss, shape [B, L].
    positions: relative position of each token, shape [B, L].
    attention_mask: input attention mask, shape [B, L].

  Returns:
    Softmax cross-entropy loss for the next-token prediction task.
  """

  # Foward pass on the input data.
  # No attention cache is needed here.
  logits, _ = model.apply(
        params,
        input_tokens,
        positions,
        None,              # Attention cache is None.
        attention_mask,
    )

  # Exclude the last step as it does not appear in the targets.
  logits = logits[0, :-1]

  # Similarly, the first token cannot be predicteds.
  target_tokens = input_tokens[0, 1:]
  target_mask = input_mask[0, 1:]

  # Convert the target labels into one-hot encoded vectors.
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

  # Don't update on unwanted tokens.
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[...,None]

  # Normalisation factor.
  norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

  # Return the nll loss.
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor
# %% [markdown]
# The Gemma transformer requires an attention mask and position vector alongside each input. We can conveniently generate these using the following function:
# %%
def get_attention_mask_and_positions(example: jax.Array,
                                     pad_id : int,
                                     )-> tuple[jax.Array, jax.Array]:
  """Builds the position and attention mask vectors from the given tokens."""
  pad_mask = example != pad_id
  current_token_position = transformer_lib.build_positions_from_mask(pad_mask)
  attention_mask = transformer_lib.make_causal_attn_mask(pad_mask)
  return current_token_position, attention_mask
# %% [markdown]
# We can now build the train_step function which performs the backward pass and updates the model's parameters accordingly.
# %%
def train_step(model: transformer_lib.Transformer,
               params,
               optimizer: optax.GradientTransformation,
               opt_state: optax.OptState,
               pad_id: int,
               example: TrainingInput):
  """Train step.

  Args:
    model: gemma transformer model.
    params: model's input parameters.
    optimizer: optax optimizer to use.
    opt_state: input optimizer's state.
    pad_id: id of the pad token.
    example: input batch.

  Returns:
    Training loss, updated parameters, updated optimizer state.
  """

  # Build the position and attention mask vectors.
  positions, attention_mask = get_attention_mask_and_positions(example.input_tokens, pad_id)

  # Forward and backward passes
  train_loss, grads = jax.value_and_grad(forward_and_loss_fn)(params,
                                                             model=model,
                                                             input_tokens=example.input_tokens,
                                                             input_mask=example.target_mask,
                                                             positions=positions,
                                                             attention_mask=attention_mask)
  # Update the parameters
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

  return train_loss, params, opt_state
# %% [markdown]
# Similarly, we build a `validation_step` function without backward pass.
# %%
def validation_step(model: transformer_lib.Transformer,
                    params,
                    pad_id: int,
                    example: TrainingInput,
                    ):
  positions, attention_mask = get_attention_mask_and_positions(example.input_tokens, pad_id)
  val_loss = forward_and_loss_fn(params,
                                 model=model,
                                 input_tokens=example.input_tokens,
                                 input_mask=example.target_mask,
                                 positions=positions,
                                 attention_mask=attention_mask)
  return val_loss
# %% [markdown]
# And now the training loop itself.
# %%
@chex.dataclass(frozen=True)
class TrainingConfig:
  learning_rate: float
  num_epochs: int
  eval_every_n: int
  batch_size: int
  max_steps: int | None = None


def train_loop(
    model: transformer_lib.Transformer,
    params,
    dataset_builder: MTNTDatasetBuilder,
    training_cfg: TrainingConfig):


  # We jit the train step, making the whole loop much more efficient
  compiled_train_step = jax.jit(train_step, static_argnames=['model', 'optimizer'])

  # We do the same with the validation step
  compiled_validation_step = jax.jit(validation_step, static_argnames=['model'])

  # To save memory, we use a SGD optimizer instead of the usual Adam. Note that
  # for this specific example SGD is more than enough.
  optimizer = optax.sgd(training_cfg.learning_rate)
  opt_state = optimizer.init(params)

  # Build the training dataset
  train_ds = dataset_builder.get_train_dataset(batch_size=training_cfg.batch_size,
                                               num_epochs=training_cfg.num_epochs)
  train_ds = train_ds.as_numpy_iterator()

  # Build the validation dataset, with a limited number of samples for this demo
  validation_ds = dataset_builder.get_validation_dataset(batch_size=training_cfg.batch_size)
  validation_ds = validation_ds.take(50)

  n_steps = 0
  avg_loss=0

  # A first round of validation loss
  n_steps_eval = 0
  eval_loss = 0
  val_iterator = validation_ds.as_numpy_iterator()
  for val_example in val_iterator:
    eval_loss += compiled_validation_step(model,
                                          params,
                                          dataset_builder._tokenizer.pad_id,
                                          val_example)
    n_steps_eval += 1
  print(f"Start, validation loss: {eval_loss/n_steps_eval}")

  for train_example in train_ds:
    train_loss, params, opt_state = compiled_train_step(model=model,
                                                        params=params,
                                                        optimizer=optimizer,
                                                        opt_state=opt_state,
                                                        pad_id=dataset_builder._tokenizer.pad_id,
                                                        example=train_example)
    n_steps += 1
    avg_loss += train_loss
    if n_steps % training_cfg.eval_every_n == 0:
      eval_loss = 0

      n_steps_eval = 0
      val_iterator = validation_ds.as_numpy_iterator()
      for val_example in val_iterator:
        eval_loss += compiled_validation_step(model,
                                              params,
                                              dataset_builder._tokenizer.pad_id,
                                              val_example)
        n_steps_eval +=1
      avg_loss /= training_cfg.eval_every_n
      eval_loss /= n_steps_eval
      print(f"STEP {n_steps} training loss: {avg_loss} - eval loss: {eval_loss}")
      avg_loss=0
    if training_cfg.max_steps is not None and n_steps > training_cfg.max_steps:
      break
  return params
# %% [markdown]
# We can fine-tune our model on a limited number of steps.
# %%
# Small seq size so that everything fits in memory
SEQ_SIZE = 25
tokenizer = GemmaTokenizer(vocab)
dataset_builder= MTNTDatasetBuilder(tokenizer, SEQ_SIZE)
training_cfg = TrainingConfig(learning_rate=1e-4,
                              num_epochs=1,
                              eval_every_n=20,
                              batch_size=1,
                              max_steps=100)

params = train_loop(model=model_2b,
                    params={'params': params['transformer']},
                    dataset_builder=dataset_builder,
                    training_cfg=training_cfg)
# %% [markdown]
# Both the training loss and the validation's are going down. But is it working ? Let's try again with our previous example:
# %%
sampler = sampler_lib.Sampler(
    transformer=model_2b,
    vocab=vocab,
    params=params['params'],
)
# %% [markdown]
# To ensure our input matches the training format, remember to use the prefix 'Translate this into French:\n'  and a newline character at the end. This signals the model to begin translation.
# %%
sampler(
    ["Translate this into French:\nHello, my name is Morgane.\n"],
    total_generation_steps=30,
    ).text
