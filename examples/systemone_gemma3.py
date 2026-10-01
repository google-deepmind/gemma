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

r"""Example: System One classification with Gemma 3 1B.

This example demonstrates fast, non-generative Jev decision primitives
(NOUL, CHOICE, SCORE) using Gemma 3 1B (`gemma3-1b-it`). All decision
questions are packed via tree attention and evaluated in a single batched
prefill pass ($O(1)$) with context-free null calibration.

Usage:

1. Run with official Gemma 3 1B IT weights (loads on CPU/GPU in ~1.5 GB):
   ```sh
   python examples/systemone_gemma3.py
   ```

2. Offline / dry-run execution with mock weights:
   ```sh
   python examples/systemone_gemma3.py --mock
   ```
"""

import argparse
import time
from typing import Any, Tuple

from gemma import gm
import jax
import jax.numpy as jnp


def setup_model_and_params(
    checkpoint: str,
    mock: bool,
) -> Tuple[gm.nn.Gemma3_1B, Any]:
  """Initializes the Gemma 3 1B model and parameters."""
  print("\n[1] Initializing Gemma3_1B model architecture...")
  model = gm.nn.Gemma3_1B()

  if mock:
    print("    [Mock Mode] Initializing model with random weights...")
    key = jax.random.key(42)
    params = model.init(key, jnp.zeros((1, 4), dtype=jnp.int32))["params"]
    print("    Mock parameters initialized.")
    return model, params

  print(f"    [Checkpoint] Loading weights from: {checkpoint}")
  print("    Downloading/streaming ~1.5 GB weights (takes ~1-2 min)...")
  start_load = time.time()
  params = gm.ckpts.load_params(checkpoint)
  print(f"    Weights loaded successfully in {time.time() - start_load:.1f}s.")
  return model, params


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Gemma 3 1B System One Decision Routing"
  )
  parser.add_argument(
      "--checkpoint",
      type=str,
      default=gm.ckpts.CheckpointPath.GEMMA3_1B_IT,
      help="Gemma 3 1B checkpoint path.",
  )
  parser.add_argument(
      "--mock",
      action="store_true",
      help="Run with random mock weights for fast offline testing.",
  )
  args = parser.parse_args()

  print("=" * 72)
  print("Gemma 3 1B IT: System One Tree-Attention Decision Routing")
  print("=" * 72)

  # 1. Initialize Gemma 3 1B model and weights
  model, params = setup_model_and_params(args.checkpoint, mock=args.mock)

  # 2. Initialize Gemma 3 Tokenizer
  print("\n[2] Initializing Gemma3Tokenizer...")
  # pylint: disable=no-value-for-parameter
  tokenizer = gm.text.Gemma3Tokenizer()
  # pylint: enable=no-value-for-parameter
  print("    Tokenizer initialized.")

  # 3. Instantiate SystemOneSampler
  sampler = gm.text.SystemOneSampler(
      model=model,
      params=params,
      tokenizer=tokenizer,
      default_temperature=1.0,
      cache_null_priors=True,
  )

  # 4. Define customer support state (shared context)
  state = (
      "Customer Message: I purchased the UltraBook Pro 14 yesterday, but upon"
      " unboxing, the display screen was completely cracked and will not turn"
      " on. I have an important client presentation tomorrow and urgently need"
      " either an immediate replacement shipped overnight or a full refund."
  )
  print("\n[3] Shared Customer Ticket State:")
  print(f"    {state}")

  # 5. Define structured decision questions (Jev primitives)
  questions = [
      gm.text.QuestionSpec(
          id="q_request_refund",
          text="Is the customer requesting a financial refund?",
          type=gm.text.QuestionType.NOUL,
      ),
      gm.text.QuestionSpec(
          id="q_issue_category",
          text="Which primary category best classifies this issue?",
          type=gm.text.QuestionType.CHOICE,
          options=[
              "Damaged Hardware",
              "Billing & Payments",
              "Account Access",
              "Software Bug",
          ],
          marginalize=True,
      ),
      gm.text.QuestionSpec(
          id="q_urgency_rating",
          text="Rate the customer urgency and dissatisfaction on a 1-5 scale.",
          type=gm.text.QuestionType.SCORE,
          score_range=(1, 5),
      ),
  ]

  print("\n[4] Setting up decision questions:")
  for q in questions:
    print(f"    * [{q.type.value.upper()}] {q.id}: {q.text}")

  # 6. Precompute null context priors for zero-shot calibration
  print("\n[5] Precomputing null context calibration priors...")
  sampler.precompute_null_priors(questions)
  print("    Null priors cached.")

  # 7. Evaluate all questions in ONE single batched tree-attention forward pass
  print("\n[6] Executing evaluate_systemone() (single tree-attention pass)...")
  response = sampler.evaluate_systemone(
      state=state,
      questions=questions,
      calibrate=True,
  )
  print(f"    Inference completed in {response.latency_ms:.2f} ms")

  # 8. Display decisions with calibrated probabilities and entropy confidence
  print("\n" + "=" * 72)
  print("System One Decision Results:")
  print("=" * 72)

  for q_id, decision in response.decisions.items():
    print(f"\n>> [{q_id}]")
    if isinstance(decision, gm.text.NoulResult):
      print("   Primitive:     NOUL (Binary Decision)")
      print(f"   Value:         {decision.value}")
      print(f"   Confidence:    {decision.confidence * 100:.2f}%")
      print(f"   Entropy:       {decision.raw_entropy:.4f} bits")
      print("   Probabilities:")
      for outcome, prob in decision.probabilities.items():
        print(f"     - {outcome!s:6s}: {prob * 100:5.2f}%")

    elif isinstance(decision, gm.text.ChoiceResult):
      print("   Primitive:     CHOICE (Categorical Routing)")
      print(f"   Selected:      [{decision.selected_index}] {decision.value}")
      print(f"   Confidence:    {decision.confidence * 100:.2f}%")
      print(f"   Entropy:       {decision.raw_entropy:.4f} bits")
      if decision.order_flip_rate is not None:
        print(f"   Order Flip Rate: {decision.order_flip_rate * 100:.1f}%")
      print("   Probabilities:")
      for opt, prob in decision.probabilities.items():
        bar = "#" * int(prob * 20)
        print(f"     - {opt:20s}: {prob * 100:5.2f}% | {bar}")

    elif isinstance(decision, gm.text.ScoreResult):
      print("   Primitive:     SCORE (Ordinal Rating)")
      print(f"   Bucket Value:  {decision.value}")
      print(f"   Expected E[S]: {decision.expected_value:.2f}")
      print(f"   Confidence:    {decision.confidence * 100:.2f}%")
      print(f"   Entropy:       {decision.raw_entropy:.4f} bits")
      print("   Probabilities:")
      for score_val, prob in decision.probabilities.items():
        bar = "#" * int(prob * 20)
        print(f"     - Score {score_val}: {prob * 100:5.2f}% | {bar}")

  print("\n" + "=" * 72)
  print("Single-pass tree-attention routing completed successfully.")
  print("=" * 72)


if __name__ == "__main__":
  main()
