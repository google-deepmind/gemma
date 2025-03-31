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
  assert out == epy.dedent("""\
      Words flow, vast and deep,
      Mimicking human insight,
      New thoughts softly bloom.
  """)
