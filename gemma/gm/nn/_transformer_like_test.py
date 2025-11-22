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

from typing import Any, ClassVar

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from gemma import gm
from gemma.gm.nn import _transformer
import jax
import jax.numpy as jnp


class MockTransformer(nn.Module):
  """Mock transformer."""

  config: gm.nn.config.TransformerConfig
  INFO: ClassVar[_transformer.ModelInfo] = _transformer.ModelInfo(
      tokenizer_version=4,
  )

  def setup(self):
    self.dense = nn.Dense(10)

  def __call__(
      self,
      tokens: jax.Array,
      *,
      images: jax.Array | None = None,
      positions: jax.Array | None = None,
      cache: gm.nn.Cache | None = None,
      attention_mask: jax.Array | None = None,
      return_last_only: bool | None = None,
      return_hidden_states: bool | None = None,
  ) -> gm.nn.Output:
    """Call the model."""
    logits = self.dense(tokens)
    return gm.nn.Output(logits=logits, cache=None, hidden_states=None)

  def init_cache(
      self,
      *,
      batch_size: int,
      dtype: jnp.dtype[Any],
      cache_length: int,
  ) -> gm.nn.Cache:
    """Initializes the KV cache for efficient generation."""
    return self.config.init_cache(
        batch_size=batch_size,
        dtype=dtype,
        cache_length=cache_length,
    )


def _init_and_apply(
    model: gm.nn.TransformerLike, tokens: jax.Array
) -> jax.Array:
  rng = jax.random.PRNGKey(0)
  params = model.init(rng, tokens)
  return model.apply(params, tokens).logits


def _get_config() -> gm.nn.config.TransformerConfig:
  return gm.nn.config.TransformerConfig(
      num_embed=13,
      embed_dim=32,
      hidden_dim=128,
      num_heads=2,
      num_kv_heads=2,
      head_dim=128,
      final_logit_softcap=None,
      attention_types=(gm.nn.AttentionType.GLOBAL,),
      use_post_attn_norm=False,
      attn_logits_soft_cap=None,
      use_post_ffw_norm=False,
  )


class TransformerLikeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='mock_model',
          get_model=lambda: MockTransformer(config=_get_config()),
          expected_shape=(1, 10),
      ),
      dict(
          testcase_name='gemma_model',
          get_model=gm.nn.Gemma3_270M,
          expected_shape=(1, 10, 262144),
      ),
  )
  def test_transformer_like(self, get_model, expected_shape):
    model = get_model()
    tokens = jnp.ones((1, 10), dtype=jnp.int32)
    output = _init_and_apply(model, tokens)
    self.assertEqual(output.shape, expected_shape)
    self.assertIsInstance(model.config, gm.nn.config.TransformerConfig)


if __name__ == '__main__':
  absltest.main()
