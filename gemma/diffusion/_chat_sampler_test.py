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
from gemma.diffusion import _chat_sampler
from gemma.diffusion import _early_stopping
from gemma.diffusion import _models
from gemma.diffusion import _sampler
from gemma.diffusion import _transformer
from gemma.gm.nn.gemma4 import _config
from gemma.gm.nn.gemma4 import _modules

_SMALL_CONFIG = _config.TransformerConfig(
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

_SMALL_SC_CONFIG = _transformer.SelfConditioningConfig(
    features=_SMALL_CONFIG.embed_dim,
    hidden_dim=_SMALL_CONFIG.hidden_dim,
)


class InterfacesTest(absltest.TestCase):

  def test_chat_sampler_instantiation(self):
    """Tests that ChatSampler can be instantiated with minimal args.

    Relying on default values for instantiation ensures that the user is not
    overwhelmed with too many required arguments.
    """
    model = _models.DiffusionGemma_26B_A4B(
        config=_SMALL_CONFIG,
        self_conditioning_config=_SMALL_SC_CONFIG,
    )

    chat = _chat_sampler.ChatSampler(model=model, params={})

    # Diffusion-specific fields
    self.assertEqual(chat.sampler.canvas_length, 256)
    self.assertEqual(chat.sampler.max_denoising_steps, 48)
    self.assertIsInstance(
        chat.sampler.diffusion_process, _sampler.DiffusionProcess
    )
    self.assertIsInstance(
        chat.sampler.diffusion_process.noise_schedule, _sampler.LinearSchedule
    )
    self.assertIsInstance(
        chat.sampler.logit_shaper, _sampler.AnnealingTemperatureShaper
    )
    self.assertEqual(chat.sampler.logit_shaper.config.exponent, 1.0)
    self.assertEqual(chat.sampler.logit_shaper.config.max_temperature, 0.8)
    self.assertEqual(chat.sampler.logit_shaper.config.min_temperature, 0.4)
    # Diffusion sampling: confidence-based selection with entropy_bound=0.1.
    self.assertIsInstance(
        chat.sampler.sample_from_predictions,
        _sampler.SampleFromPredictions,
    )
    self.assertEqual(chat.sampler.sample_from_predictions.entropy_bound, 0.1)
    self.assertIsInstance(
        chat.sampler.early_stop_fn, _early_stopping.ChainedEarlyStop
    )
    self.assertLen(chat.sampler.early_stop_fn.early_stop_fns, 2)
    self.assertIsInstance(
        chat.sampler.early_stop_fn.early_stop_fns[0],
        _early_stopping.TokenStabilityEarlyStop,
    )
    self.assertIsInstance(
        chat.sampler.early_stop_fn.early_stop_fns[1],
        _early_stopping.EntropyEarlyStop,
    )
    self.assertEqual(
        chat.sampler.early_stop_fn.early_stop_fns[1].entropy_threshold,
        0.005,
    )

  def test_chat_sampler_instantiation_overrides(self):
    """Tests that ChatSampler can be instantiated with optional args."""

    model = _models.DiffusionGemma_26B_A4B(
        config=_SMALL_CONFIG,
        self_conditioning_config=_SMALL_SC_CONFIG,
    )

    chat = _chat_sampler.ChatSampler(
        model=model,
        params={},
        canvas_length=32,
        max_denoising_steps=16,
        logit_shaper=_sampler.AnnealingTemperatureShaper(
            config=_sampler.AnnealingTemperatureShaperConfig(
                max_temperature=0.9,
                min_temperature=0.3,
                exponent=2.0,
            )
        ),
    )

    # Confirm that the overrides are applied (i.e. they are different from the
    # defaults)
    self.assertEqual(chat.sampler.canvas_length, 32)
    self.assertEqual(chat.sampler.max_denoising_steps, 16)
    self.assertEqual(chat.sampler.logit_shaper.config.max_temperature, 0.9)
    self.assertEqual(chat.sampler.logit_shaper.config.min_temperature, 0.3)
    self.assertEqual(chat.sampler.logit_shaper.config.exponent, 2.0)


if __name__ == "__main__":
  absltest.main()
