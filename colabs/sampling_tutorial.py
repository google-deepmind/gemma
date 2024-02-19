# This is a Colab notebook. Consider opening in http://colab.google/ or as
# a notebook in JupyterLab.
# %% [markdown]
# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#
# ---
# %% [markdown]
# # Getting Started with Gemma Sampling: A Step-by-Step Guide
#
# You will find in this colab a detailed tutorial explaining how to load a Gemma checkpoint and sample from it.
#

# %% [markdown]
# ## Setup
#
# Please follow installation instructions at https://github.com/google-deepmind/gemma/blob/main/README.md.
# %% Download the checkpoints
# Download the Flax checkpoints from https://www.kaggle.com/models/google/gemma
# and put the local paths below.

ckpt_path = ''
vocab_path = ''

# %% Python imports
from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib
import sentencepiece as spm
# %% [markdown]
# ## Start Generating with Your Model
#
# Load and prepare your LLM's checkpoint for use with Flax.
# %%
# Load parameters
params = params_lib.load_and_format_params(ckpt_path)
# %% [markdown]
# Load your tokenizer, which we'll construct using the [SentencePiece](https://github.com/google/sentencepiece) library.
# %%
vocab = spm.SentencePieceProcessor()
vocab.Load(vocab_path)
# %% [markdown]
# Use the `transformer_lib.TransformerConfig.from_params` function to automatically load the correct configuration from a checkpoint. Note that the vocabulary size is smaller than the number of input embeddings due to unused tokens in this release.
# %%
transformer_config=transformer_lib.TransformerConfig.from_params(
    params,
    cache_size=1024  # Number of time steps in the transformer's cache
)
transformer = transformer_lib.Transformer(transformer_config)
# %% [markdown]
# Finally, build a sampler on top of your model and your tokenizer.
# %%
# Create a sampler with the right param shapes.
sampler = sampler_lib.Sampler(
    transformer=transformer,
    vocab=vocab,
    params=params['transformer'],
)
# %% [markdown]
# You're ready to start sampling ! This sampler uses just-in-time compilation, so changing the input shape triggers recompilation, which can slow things down. For the fastest and most efficient results, keep your batch size consistent.
# %%
input_batch = [
    "\n# Python program for implementation of Bubble Sort\n\ndef bubbleSort(arr):",
    "What are the planets of the solar system?",
  ]

out_data = sampler(
    input_strings=input_batch,
    total_generation_steps=300,  # number of steps performed when generating
  )

for input_string, out_string in zip(input_batch, out_data.text):
  print(f"Prompt:\n{input_string}\nOutput:\n{out_string}")
  print()
  print(10*'#')
# %% [markdown]
# You should get an implementation of bubble sort and a description of the solar system.
