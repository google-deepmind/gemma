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

from absl.testing import absltest
from gemma.diffusion import _models
from gemma.diffusion import _transformer
from gemma.gm.nn.gemma4 import _config
from gemma.gm.nn.gemma4 import _modules
import jax
import jax.numpy as jnp


class DiffusionGemmaTest(absltest.TestCase):

  def test_can_instantiate_from_default_config(self):
    """Tests that DiffusionGemma_A26B_A4B can be instantiated without any args."""
    model = _models.DiffusionGemma_A26B_A4B()
    self.assertIsNotNone(model)

  def test_multiple_token_generation(self):
    batch_size = 1
    cache_length = 10

    small_config = _config.TransformerConfig(
        num_embed=32,
        embed_dim=8,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        hidden_dim=16,
        attention_types=[_modules.AttentionType.GLOBAL],
        kv_cache_sharing_config=None,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcap=None,
        global_rope_proportion=1.0,
    )

    model = _models.DiffusionGemma_A26B_A4B(
        config=small_config,
        self_conditioning_config=_transformer.SelfConditioningConfig(
            features=small_config.embed_dim,
            hidden_dim=small_config.hidden_dim,
        ),
    )

    rng = jax.random.PRNGKey(0)
    fake_tokens = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    fake_sc_embeddings = jnp.zeros(
        (batch_size, 1, small_config.embed_dim), dtype=jnp.float32
    )
    variables = model.init(
        rng,
        tokens=fake_tokens,
        sc_embeddings=fake_sc_embeddings,
        method=model.call_with_self_conditioning,
    )

    # Init Cache
    cache = model.init_cache(
        batch_size=batch_size, dtype=jnp.float32, cache_length=cache_length
    )

    # Create input: batch=1, seq_len=4
    input_tokens = jnp.ones((batch_size, 4), dtype=jnp.int32)

    # Create attention mask
    # Shape [batch, seq_len, cache_len]
    attention_mask = jnp.ones((batch_size, 4, cache_length), dtype=jnp.bool_)

    # Run forward with 4 tokens
    sc_embeddings = jnp.zeros(
        (batch_size, 4, small_config.embed_dim), dtype=jnp.float32
    )

    output = model.apply(
        variables,
        tokens=input_tokens,
        sc_embeddings=sc_embeddings,
        cache=cache,
        attention_mask=attention_mask,
        method=model.call_with_self_conditioning,
    )

    # Check output logits shape: [batch, 4, num_embed]
    self.assertEqual(output.logits.shape, (batch_size, 4, 32))

    # Check cache update
    # New end index should be 4 (0 + 4)
    new_cache = output.cache
    self.assertEqual(new_cache["layer_0"]["end_index"][0], 4)


if __name__ == "__main__":
  absltest.main()
