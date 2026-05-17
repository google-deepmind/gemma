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

from gemma.gm.nn.gemma4.audio import _model as audio_model
from gemma.gm.nn.gemma4.audio import _modules as audio_modules
import jax
import jax.numpy as jnp


def test_model_initialization_and_shape():
  # 1. Initialize the model config
  config = audio_modules.ConformerConfig(
      num_layers=2,  # Use fewer layers for faster test
      model_dims=128,
      lm_model_dims=256,
      atten_num_heads=4,
  )
  model = audio_model.AudioTokenizer(config=config)

  # 2. Create a random speech signal
  batch_size = 2
  # 1 second of audio at 16kHz
  num_samples = 16000
  rng = jax.random.PRNGKey(0)
  x_rng, _, init_rng = jax.random.split(rng, 3)

  # Input shape expected by __call__ is [batch_size, 1, num_samples]
  # but it reshapes it internally to that.
  # Actually, the code says:
  # batch_size = x.shape[0]
  # x = x.reshape(batch_size, 1, -1)
  # So we can pass [batch_size, num_samples]
  x = jax.random.normal(x_rng, (batch_size, num_samples))

  # 3. Initialize model parameters
  sequence_lengths = jnp.full((batch_size,), num_samples, dtype=jnp.int32)
  variables = model.init(init_rng, x, sequence_lengths)  # jax.eval_shape...

  # 4. Check the output shape
  output, padding_mask = model.apply(variables, x, sequence_lengths)

  # Gemax_MelFilterbank produces spectrogram with hop_length=160.
  # num_frames = (num_samples + hop_length - 1) // hop_length
  # For 16000 samples, hop 160, it should be 100 frames.
  # SubSamplingBlock reduces seq_len by factor of 4 (two layers with stride 2)
  # expected_seq_len = 100 // 4 = 25.

  expected_seq_len = 25
  expected_shape = (batch_size, expected_seq_len, config.lm_model_dims)

  assert output.shape == expected_shape
  assert padding_mask.shape == (batch_size, expected_seq_len)
  assert padding_mask.dtype == jnp.bool_
  assert not jnp.any(jnp.isnan(output))
