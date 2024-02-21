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
# ============================================================================
"""Minimal test for sampler."""

import os
import unittest

from absl.testing import absltest
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib
import jax
import jax.numpy as jnp
import numpy as np

import sentencepiece as spm

# TODO: Replace with a mock tokenizer.
# Download the tokenizer and put its path here.
_VOCAB = ''
_NO_TOKENIZER_MESSAGE = (
    'No tokenizer path specified. Please download a tokenizer and update the'
    ' `_VOCAB` constant.'
)


class SamplerTest(absltest.TestCase):

  @unittest.skipIf(not _VOCAB, _NO_TOKENIZER_MESSAGE)
  def test_samples(self):
    vocab = spm.SentencePieceProcessor()
    vocab.Load(_VOCAB)

    transformer_config = transformer_lib.TransformerConfig(
        num_layers=6,
        num_embed=vocab.GetPieceSize() + 128,  # 128 for padding
        embed_dim=768,
        hidden_dim=6144,
        num_heads=4,
        num_kv_heads=4,
        head_dim=256,
        max_cache_length=1024,
    )
    attention_mask = jnp.ones((1, 1, transformer_config.max_cache_length))
    cache = transformer_lib.init_cache(
        transformer_config, 1, dtype=jnp.float32
    )
    transformer = transformer_lib.Transformer(transformer_config)
    params = transformer.init(
        jax.random.PRNGKey(0),
        jnp.array([[1]]),
        jnp.array([[1]]),
        cache,
        attention_mask,
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
        params=params['params'],
        dtype=jnp.float32,
    )

    result = sampler(['input string', 'hello world'], total_generation_steps=10)
    self.assertIsNotNone(result)

  @unittest.skipIf(not _VOCAB, _NO_TOKENIZER_MESSAGE)
  def test_forward_equivalence(self):
    vocab = spm.SentencePieceProcessor()
    vocab.Load(_VOCAB)
    transformer_config = transformer_lib.TransformerConfig(
        num_layers=2,
        num_embed=vocab.GetPieceSize() + 128,  # 128 for padding
        embed_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_kv_heads=1,
        head_dim=64,
        max_cache_length=8,
    )

    transformer = transformer_lib.Transformer(transformer_config)
    raw_input = 'Hello ! My name is Morgane'
    token_input = jnp.asarray(
        [vocab.bos_id()] + vocab.EncodeAsIds(raw_input)
    ).reshape((1, -1))
    batch_size = 1
    cache = transformer_lib.init_cache(
        transformer_config, batch_size, dtype=jnp.float32
    )
    input_mask = token_input != vocab.pad_id()
    positions = transformer_lib.build_positions_from_mask(
        input_mask
    )
    attention_mask = transformer_lib.make_causal_attn_mask(
        token_input != vocab.pad_id()
    )

    n_input_tokens = token_input.shape[1]

    params = transformer.init(
        jax.random.PRNGKey(42),
        token_input,
        positions,
        cache,
        attention_mask,
    )

    output_forward, _ = transformer.apply(
        params,
        last_tokens=token_input,
        positions=positions,
        cache=cache,
        attention_mask=attention_mask,
    )
    output_forward = output_forward[0, :n_input_tokens]

    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
        params=params['params'],
        dtype=jnp.float32,
    )

    output_transformer = sampler(
        [raw_input], total_generation_steps=10, echo=True
    )
    out_logits = np.array(output_transformer.logits)[0, 1 : n_input_tokens + 1]

    np.testing.assert_almost_equal(output_forward, out_logits)


if __name__ == '__main__':
  absltest.main()
