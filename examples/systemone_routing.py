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

r"""Example: Zero-shot System One classification and tree-attention routing.

This script demonstrates using `SystemOneSampler` to perform fast,
non-generative heuristic triage (Jev decision primitives) in a single
batched prefill pass ($O(1)$) without any autoregressive generation loops.

Run with:

```sh
python examples/systemone_routing.py
```
"""

from gemma import gm
from gemma.gm.nn import _transformer
from gemma.gm.nn import config as config_lib
import jax
import jax.numpy as jnp


class DemoGemma(_transformer.Transformer):
  """Self-contained Gemma configuration for local demo execution."""

  config: config_lib.TransformerConfig = config_lib.TransformerConfig(
      num_embed=100,
      embed_dim=64,
      hidden_dim=128,
      num_heads=4,
      num_kv_heads=4,
      head_dim=32,
      final_logit_softcap=None,
      attention_types=(config_lib.AttentionType.GLOBAL,),
      use_post_attn_norm=None,
      attn_logits_soft_cap=None,
      use_post_ffw_norm=None,
  )


class SimpleDemoTokenizer:
  """Demo tokenizer mapping prompt vocabulary to token IDs."""

  def __init__(self):
    self.bos_id = 1
    self.eos_id = 2
    self.pad_id = 0
    self.vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "Context:": 3,
        "Question:": 4,
        "Answer:": 5,
        "True": 6,
        "False": 7,
        "A": 10,
        "B": 11,
        "C": 12,
        "D": 13,
        "1": 21,
        "2": 22,
        "3": 23,
        "4": 24,
        "5": 25,
    }

  def encode(self, text: str, add_bos: bool = False) -> list[int]:
    words = text.replace("\n", " ").split()
    tokens = []
    for w in words:
      clean_w = w.strip(".,;:?!")
      if clean_w in self.vocab:
        tokens.append(self.vocab[clean_w])
      elif w in self.vocab:
        tokens.append(self.vocab[w])
      else:
        tokens.append(30 + (hash(w) % 50))
    if not tokens:
      tokens = [30]
    if add_bos:
      tokens = [self.bos_id] + tokens
    return tokens


def main():
  print("=" * 70)
  print("Gemma SystemOneSampler: Tree-Attention Jev Decision Routing Demo")
  print("=" * 70)

  # 1. Initialize model, weights, and tokenizer
  print("\n[1] Initializing Gemma model and tokenizer...")
  model = DemoGemma()
  key = jax.random.PRNGKey(42)
  params = model.init(key, jnp.zeros((1, 4), dtype=jnp.int32))["params"]
  tokenizer = SimpleDemoTokenizer()

  sampler = gm.text.SystemOneSampler(
      model=model,
      params=params,
      tokenizer=tokenizer,
      default_temperature=1.0,
      cache_null_priors=True,
  )
  print("    Model and SystemOneSampler initialized successfully.")

  # 2. Define customer-service triage state (shared context)
  state = (
      "Customer message: I ordered the Nova Ultra 15 laptop on Monday"
      " (Order #NW-98231). The laptop screen arrived completely shattered, and"
      " the device will not turn on. I have an urgent business trip this Friday"
      " and cannot work without it. I need an immediate full refund or a"
      " replacement overnighted to my address today."
  )
  print("\n[2] Shared Customer Service State:")
  print(f"    {state}")

  # 3. Define structured Jev decision primitives
  print("\n[3] Setting up decision questions:")
  questions = [
      gm.text.QuestionSpec(
          id="q_refund_risk",
          text="Is this user requesting a monetary refund?",
          type=gm.text.QuestionType.NOUL,
      ),
      gm.text.QuestionSpec(
          id="q_dept_routing",
          text="Which department should handle this ticket?",
          type=gm.text.QuestionType.CHOICE,
          options=[
              "Hardware Replacement",
              "Billing & Refunds",
              "Technical Support",
              "General Inquiry",
          ],
      ),
      gm.text.QuestionSpec(
          id="q_urgency_score",
          text="Rate the customer urgency and frustration on a 1-5 scale.",
          type=gm.text.QuestionType.SCORE,
          score_range=(1, 5),
      ),
  ]
  for q in questions:
    print(f"    * [{q.type.value.upper()}] {q.id}: {q.text}")

  # 4. Evaluate all questions in ONE single batched tree-attention forward pass
  print("\n[4] Executing evaluate_systemone() (1 forward pass prefill)...")
  response = sampler.evaluate_systemone(state=state, questions=questions)
  print(f"    Completed in {response.latency_ms:.2f} ms")

  # 5. Display structured decisions
  print("\n[5] Triage Results:")
  for q_id, decision in response.decisions.items():
    print(f"\n--- Decision: {q_id} ---")
    if isinstance(decision, gm.text.NoulResult):
      print("    Type:        NOUL (Binary)")
      print(f"    Value:       {decision.value}")
      print(f"    Confidence:  {decision.confidence * 100:.2f}%")
      print(f"    Probabilities: {decision.probabilities}")
      print(f"    Entropy:     {decision.raw_entropy:.4f} bits")

    elif isinstance(decision, gm.text.ChoiceResult):
      print("    Type:        CHOICE (Categorical)")
      print(f"    Selected:    [{decision.selected_index}] {decision.value}")
      print(f"    Confidence:  {decision.confidence * 100:.2f}%")
      print("    Probabilities:")
      for opt, p in decision.probabilities.items():
        bar = "#" * int(p * 20)
        print(f"      - {opt:22s}: {p * 100:5.1f}% | {bar}")
      print(f"    Entropy:     {decision.raw_entropy:.4f} bits")

    elif isinstance(decision, gm.text.ScoreResult):
      print("    Type:        SCORE (Ordinal Rating)")
      print(f"    Bucket Val:  {decision.value}")
      print(f"    Expected:    {decision.expected_value:.2f}")
      print(f"    Confidence:  {decision.confidence * 100:.2f}%")
      print("    Probabilities:")
      for score_val, p in decision.probabilities.items():
        bar = "#" * int(p * 20)
        print(f"      - Score {score_val}: {p * 100:5.1f}% | {bar}")
      print(f"    Entropy:     {decision.raw_entropy:.4f} bits")

  print("\n" + "=" * 70)
  print(
      "Execution successful: All 3 decisions resolved in a single prefill pass."
  )
  print("=" * 70)


if __name__ == "__main__":
  main()
