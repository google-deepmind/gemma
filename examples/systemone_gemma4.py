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

r"""Example: System One classification with Gemma 4.

This example demonstrates fast, non-generative Jev decision primitives
(NOUL, CHOICE, SCORE) using the Gemma 4 architecture and tokenizer. All
decision questions are packed via tree attention and evaluated in a single
batched prefill pass ($O(1)$) with context-free null calibration.

Usage:

1. Lightweight local execution (runs on CPU without large weights):
   ```sh
   python examples/systemone_gemma4.py --mock
   ```

2. Full Gemma 4 E2B checkpoint execution (requires memory/download):
   ```sh
   python examples/systemone_gemma4.py \
       --checkpoint gs://gemma-data/checkpoints/gemma4-e2b-it
   ```
"""

import argparse
from typing import Any, Tuple

from gemma import gm
from gemma.gm.nn import _transformer
from gemma.gm.nn import config as config_lib
import jax
import jax.numpy as jnp


class TinyGemma4(_transformer.Transformer):
  """Lightweight Gemma 4 compatible model for fast local CPU execution."""

  config: config_lib.TransformerConfig = config_lib.TransformerConfig(
      num_embed=262144,  # Gemma 4 vocab size matching Gemma4Tokenizer
      embed_dim=64,
      hidden_dim=128,
      num_heads=4,
      num_kv_heads=1,
      head_dim=32,
      final_logit_softcap=30.0,
      attention_types=(config_lib.AttentionType.GLOBAL,),
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
  )
  INFO = _transformer.ModelInfo(tokenizer_version=4)


def setup_model_and_params(
    checkpoint: str | None,
    mock: bool,
) -> Tuple[Any, Any]:
  """Initializes the Gemma 4 model and parameters."""
  if checkpoint and not mock:
    print(f"\n[Model] Loading official Gemma 4 E2B checkpoint: {checkpoint}")
    print("        (Smallest Gemma 4 checkpoint, text_only=True)")
    model = gm.nn.Gemma4_E2B(text_only=True)
    params = gm.ckpts.load_params(checkpoint, text_only=True)
    print("        Checkpoint weights loaded successfully.")
    return model, params

  print("\n[Model] Using lightweight Gemma 4 compatible model (CPU-friendly).")
  print("        Pass --checkpoint to load full Gemma 4 E2B weights.")
  model = TinyGemma4()
  key = jax.random.key(42)
  params = model.init(key, jnp.zeros((1, 4), dtype=jnp.int32))["params"]
  print("        Lightweight model initialized successfully.")
  return model, params


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Gemma 4 System One Decision Routing"
  )
  parser.add_argument(
      "--checkpoint",
      type=str,
      default=None,
      help=(
          "Gemma 4 checkpoint path (e.g."
          f" '{gm.ckpts.CheckpointPath.GEMMA4_E2B_IT}')."
      ),
  )
  parser.add_argument(
      "--mock",
      action="store_true",
      help="Force lightweight model for fast local testing on CPU.",
  )
  args = parser.parse_args()

  # If no checkpoint is given, default to mock mode for safe local run.
  use_mock = args.mock or (args.checkpoint is None)

  print("=" * 72)
  print("Gemma 4 System One Classifier: Tree-Attention Decision Routing")
  print("=" * 72)

  # 1. Initialize Gemma 4 model and params
  model, params = setup_model_and_params(args.checkpoint, mock=use_mock)

  # 2. Initialize Gemma 4 Tokenizer
  print("\n[Tokenizer] Initializing Gemma4Tokenizer...")
  # pylint: disable=no-value-for-parameter
  tokenizer = gm.text.Gemma4Tokenizer()
  # pylint: enable=no-value-for-parameter
  print("            Tokenizer initialized.")

  # 3. Instantiate SystemOneSampler
  sampler = gm.text.SystemOneSampler(
      model=model,
      params=params,
      tokenizer=tokenizer,
      default_temperature=1.0,
      cache_null_priors=True,
  )

  # 4. Define customer incident report (shared context state)
  state = (
      "Incident Alert: High latency and 504 Gateway Timeouts reported on"
      " payment service 'pay-gateway-us-east' starting at 14:22 UTC. Over 15%"
      " of incoming checkout transactions are failing. Database CPU utilization"
      " is at 98% with multiple locked transaction threads."
  )
  print("\n[Context] Shared Incident State:")
  print(f"    {state}")

  # 5. Define structured decision questions (Jev primitives)
  questions = [
      gm.text.QuestionSpec(
          id="q_page_oncall",
          text=(
              "Does this incident require immediately paging on-call engineers?"
          ),
          type=gm.text.QuestionType.NOUL,
      ),
      gm.text.QuestionSpec(
          id="q_routing_team",
          text="Which engineering team is the primary owner for this incident?",
          type=gm.text.QuestionType.CHOICE,
          options=[
              "Database Infrastructure",
              "Frontend Checkout",
              "Network Operations",
              "Security & Compliance",
          ],
      ),
      gm.text.QuestionSpec(
          id="q_severity_rating",
          text="Rate the incident severity on an ITIL scale of 1 to 5.",
          type=gm.text.QuestionType.SCORE,
          score_range=(1, 5),
      ),
  ]

  print("\n[Questions] Decision primitives to resolve:")
  for q in questions:
    print(f"    * [{q.type.value.upper()}] {q.id}: {q.text}")

  # 6. Precompute null priors for zero-shot context-free calibration
  print("\n[Calibration] Precomputing null context priors...")
  sampler.precompute_null_priors(questions)
  print("              Null priors cached.")

  # 7. Evaluate all questions in a single tree-attention prefill pass ($O(1)$)
  print("\n[Inference] Executing single-pass tree-attention forward pass...")
  response = sampler.evaluate_systemone(
      state=state,
      questions=questions,
      calibrate=True,
  )
  print(f"            Completed in {response.latency_ms:.2f} ms")

  # 8. Display decisions with confidence and calibrated probabilities
  print("\n" + "=" * 72)
  print("System One Decision Results:")
  print("=" * 72)

  for q_id, decision in response.decisions.items():
    print(f"\n>> [{q_id}]")
    if isinstance(decision, gm.text.NoulResult):
      print("   Primitive:   NOUL (Binary Decision)")
      print(f"   Value:       {decision.value}")
      print(f"   Confidence:  {decision.confidence * 100:.2f}%")
      print(f"   Entropy:     {decision.raw_entropy:.4f} bits")
      print("   Probabilities:")
      for outcome, prob in decision.probabilities.items():
        print(f"     - {outcome!s:6s}: {prob * 100:5.1f}%")

    elif isinstance(decision, gm.text.ChoiceResult):
      print("   Primitive:   CHOICE (Categorical Routing)")
      print(f"   Selected:    [{decision.selected_index}] {decision.value}")
      print(f"   Confidence:  {decision.confidence * 100:.2f}%")
      print(f"   Entropy:     {decision.raw_entropy:.4f} bits")
      print("   Probabilities:")
      for opt, prob in decision.probabilities.items():
        bar = "#" * int(prob * 20)
        print(f"     - {opt:25s}: {prob * 100:5.1f}% | {bar}")

    elif isinstance(decision, gm.text.ScoreResult):
      print("   Primitive:   SCORE (Ordinal Rating)")
      print(f"   Bucket Val:  {decision.value}")
      print(f"   Expected E:  {decision.expected_value:.2f}")
      print(f"   Confidence:  {decision.confidence * 100:.2f}%")
      print(f"   Entropy:     {decision.raw_entropy:.4f} bits")
      print("   Probabilities:")
      for score_val, prob in decision.probabilities.items():
        bar = "#" * int(prob * 20)
        print(f"     - Score {score_val}: {prob * 100:5.1f}% | {bar}")

  print("\n" + "=" * 72)
  print("Single-pass tree-attention routing completed successfully.")
  print("=" * 72)


if __name__ == "__main__":
  main()
