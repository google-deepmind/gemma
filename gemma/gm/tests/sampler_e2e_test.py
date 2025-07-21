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

"""End-to-end test for the sampler."""

from etils import epy
from gemma import gm


def test_sampler():

  # Model and parameters
  model = gm.nn.Gemma3_4B()
  params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

  # Example of multi-turn conversation
  sampler = gm.text.ChatSampler(
      model=model,
      params=params,
  )
  out = sampler.chat('Write a haiku about LLMs. No comments.')
  assert out == epy.dedent("""
      Words flow, vast and deep,
      Mimicking human insight,
      New thoughts softly bloom.
  """)

  out = sampler.chat('Share one metaphor linking "shadow" and "laughter".')
  assert out == epy.dedent("""
    Here’s a metaphor linking “shadow” and “laughter”:

    **“Laughter is the sunlight that chases away a shadow, but the shadow always lingers, a quiet reminder of the darkness it once held.”**

    **Explanation:**

    *   **Laughter** represents the bright, joyful moments that dispel sadness or worry.
    *   **Shadow** symbolizes the lingering feelings of sadness, fear, or past experiences that can still be present, even when happiness is felt. 

    The metaphor suggests that while laughter can temporarily banish negativity, the underlying potential for darkness (the shadow) remains, subtly present beneath the surface. 

    ---

    Would you like me to try another metaphor, or perhaps explore this idea in more detail?
  """)


def test_sampler_stop_token():

  # Make sure the model stops after producing the token "laughter".
  stop_token = 'laughter'

  # Model and parameters
  model = gm.nn.Gemma3_4B()
  params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

  # Example of multi-turn conversation
  sampler = gm.text.ChatSampler(
      model=model,
      params=params,
      stop_tokens=[stop_token],
  )
  out = sampler.chat('Share one metaphor linking "shadow" and "laughter".')
  assert out == 'Here’s a metaphor linking “shadow” and “laughter'
