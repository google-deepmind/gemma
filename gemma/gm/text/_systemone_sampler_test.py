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

"""Unit tests for SystemOneSampler and Jev decision primitives."""

# pylint: disable=protected-access,not-callable,unused-variable

import math
from unittest import mock

from absl.testing import absltest
from gemma import gm
from gemma.gm.nn import _transformer
from gemma.gm.nn import config as config_lib
import jax
import jax.numpy as jnp
import numpy as np


class MockTokenizer:
  """Mock tokenizer for unit testing."""

  def __init__(self):
    self.bos_id = 1
    self.eos_id = 2
    self.pad_id = 0
    # Vocabulary mapping tokens to IDs
    self.token_to_id = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "Context:": 3,
        "Question:": 4,
        "Answer:": 5,
        "with": 6,
        "True": 7,
        "or": 8,
        "False.": 9,
        "False": 10,
        "A": 11,
        "B": 12,
        "C": 13,
        "1": 21,
        "2": 22,
        "3": 23,
        "4": 24,
        "5": 25,
        "state": 30,
        "q1": 31,
        "q2": 32,
        "q3": 33,
        "multi": 40,
        "piece": 41,
    }
    self.id_to_token = {v: k for k, v in self.token_to_id.items()}

  def encode(self, text: str, add_bos: bool = False) -> list[int]:
    # Custom rule for testing multi-token failure
    if text.strip() == "multitoken":
      tokens = [40, 41]
    else:
      words = text.replace("\n", " ").split()
      tokens = []
      for w in words:
        if w in self.token_to_id:
          tokens.append(self.token_to_id[w])
        else:
          # Unknown token mapped deterministically to a valid id in [50, 99]
          tokens.append(50 + (hash(w) % 40))

    if not tokens:
      tokens = [30]

    if add_bos:
      tokens = [self.bos_id] + tokens
    return tokens


class TinyMockGemma(_transformer.Transformer):
  """Lightweight CPU Gemma model for deterministic testing."""

  config: config_lib.TransformerConfig = config_lib.TransformerConfig(
      num_embed=100,
      embed_dim=32,
      hidden_dim=64,
      num_heads=2,
      num_kv_heads=2,
      head_dim=32,
      final_logit_softcap=None,
      attention_types=(config_lib.AttentionType.GLOBAL,),
      use_post_attn_norm=None,
      attn_logits_soft_cap=None,
      use_post_ffw_norm=None,
  )


class SystemOneSamplerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tokenizer = MockTokenizer()
    self.model = TinyMockGemma()
    key = jax.random.PRNGKey(42)
    self.params = self.model.init(key, jnp.zeros((1, 4), dtype=jnp.int32))[
        "params"
    ]
    self.sampler = gm.text.SystemOneSampler(
        model=self.model,
        params=self.params,
        tokenizer=self.tokenizer,
        default_temperature=1.0,
    )

  def test_tree_attention_pack_structure(self):
    """Verifies tree packing inputs, positions, masks, and terminal indices."""
    state_ids = [1, 30]
    q_ids_1 = [4, 31, 5]
    q_ids_2 = [4, 32, 5]

    input_ids, positions, attention_mask, terminal_indices = (
        gm.text.build_tree_attention_pack(state_ids, [q_ids_1, q_ids_2])
    )

    total_len = len(state_ids) + len(q_ids_1) + len(q_ids_2)
    self.assertEqual(input_ids.shape, (1, total_len))
    self.assertEqual(positions.shape, (1, total_len))
    self.assertEqual(attention_mask.shape, (1, 1, total_len, total_len))
    self.assertEqual(len(terminal_indices), 2)

    # Position offsets: Q1 and Q2 both restart RoPE position at len(state_ids)=2
    expected_positions = [0, 1, 2, 3, 4, 2, 3, 4]
    np.testing.assert_array_equal(positions[0], expected_positions)

    # Terminal indices should point to the end of each branch
    self.assertEqual(terminal_indices[0], 4)  # 2 + 3 - 1
    self.assertEqual(terminal_indices[1], 7)  # 2 + 3 + 3 - 1

    # Check 2D mask properties
    mask_2d = np.array(attention_mask[0, 0])
    # State cannot attend to question branches
    self.assertFalse(mask_2d[0, 2:].any())
    self.assertFalse(mask_2d[1, 2:].any())
    # Q1 can attend to state and itself, but NOT Q2
    self.assertTrue(mask_2d[2:5, :2].all())
    self.assertFalse(mask_2d[2:5, 5:].any())
    # Q2 can attend to state and itself, but NOT Q1
    self.assertTrue(mask_2d[5:, :2].all())
    self.assertFalse(mask_2d[5:, 2:5].any())

  def test_tree_attention_max_seq_len_guard(self):
    """Verifies that exceeding max_seq_len raises ValueError."""
    state_ids = [1, 2, 3]
    q_ids = [[4, 5, 6]]
    with self.assertRaises(ValueError):
      gm.text.build_tree_attention_pack(state_ids, q_ids, max_seq_len=5)

  def test_calibrate_and_score_probabilities_sum(self):
    """Asserts probabilities strictly sum to 1.0 +- 1e-6."""
    real_logits = jnp.array([2.5, -1.2, 0.4, 3.1])
    probs, entropy, confidence = gm.text.calibrate_and_score(real_logits)
    self.assertAlmostEqual(float(jnp.sum(probs)), 1.0, places=6)
    self.assertGreaterEqual(confidence, 0.0)
    self.assertLessEqual(confidence, 1.0)

  def test_calibrate_and_score_equal_logits(self):
    """Asserts equal logits yield confidence == 0.0."""
    real_logits = jnp.array([1.5, 1.5, 1.5, 1.5])
    probs, entropy, confidence = gm.text.calibrate_and_score(real_logits)
    self.assertAlmostEqual(confidence, 0.0, places=6)
    expected_entropy = math.log2(4)
    self.assertAlmostEqual(entropy, expected_entropy, places=6)

  def test_calibrate_and_score_one_hot_logits(self):
    """Asserts one-hot logits yield confidence == 1.0."""
    real_logits = jnp.array([1000.0, -1000.0])
    probs, entropy, confidence = gm.text.calibrate_and_score(real_logits)
    self.assertAlmostEqual(confidence, 1.0, places=6)
    self.assertAlmostEqual(entropy, 0.0, places=6)
    self.assertAlmostEqual(float(probs[0]), 1.0, places=6)
    self.assertAlmostEqual(float(probs[1]), 0.0, places=6)

  def test_multi_token_candidates_raise_value_error(self):
    """Asserts candidate tokenizing to multiple tokens raises ValueError."""
    with self.assertRaises(ValueError):
      gm.text._systemone_sampler._resolve_candidate_tokens(
          self.tokenizer, ["A", "multitoken"]
      )

    # Test when a question primitive's candidate tokenizes to multiple tokens
    class MultiTokenTokenizer(MockTokenizer):

      def encode(self, text: str, add_bos: bool = False) -> list[int]:
        if "True" in text:
          return [7, 8]  # Multi-token True
        return super().encode(text, add_bos=add_bos)

    bad_sampler = gm.text.SystemOneSampler(
        model=self.model,
        params=self.params,
        tokenizer=MultiTokenTokenizer(),
    )
    with self.assertRaises(ValueError):
      bad_sampler.evaluate_noul(state="state", question="test", calibrate=False)

  def test_tree_attention_parity_with_sequential(self):
    """Asserts tree attention matches sequential evaluation within eps."""
    state = "state"
    q_noul = gm.text.QuestionSpec(
        id="q1",
        text="q1",
        type=gm.text.QuestionType.NOUL,
    )
    q_choice = gm.text.QuestionSpec(
        id="q2",
        text="q2",
        type=gm.text.QuestionType.CHOICE,
        options=["A", "B", "C"],
    )

    # 1. Sequential evaluations
    seq_noul = self.sampler.evaluate_noul(
        state=state, question=q_noul.text, question_id="q1", calibrate=False
    )
    seq_choice = self.sampler.evaluate_choice(
        state=state,
        question=q_choice.text,
        options=q_choice.options,
        question_id="q2",
        calibrate=False,
    )

    # 2. Parallel tree attention evaluation
    tree_resp = self.sampler.evaluate_systemone(
        state=state, questions=[q_noul, q_choice], calibrate=False
    )

    tree_noul = tree_resp.decisions["q1"]
    tree_choice = tree_resp.decisions["q2"]

    # Verify probability distributions match closely
    for k in seq_noul.probabilities:
      self.assertAlmostEqual(
          tree_noul.probabilities[k], seq_noul.probabilities[k], delta=1e-4
      )
    for k in seq_choice.probabilities:
      self.assertAlmostEqual(
          tree_choice.probabilities[k], seq_choice.probabilities[k], delta=1e-4
      )

    self.assertAlmostEqual(
        tree_noul.confidence, seq_noul.confidence, delta=1e-4
    )
    self.assertAlmostEqual(
        tree_choice.confidence, seq_choice.confidence, delta=1e-4
    )

  def test_no_inter_question_contamination(self):
    """Asserts changing Q2 text leaves Q1 logits completely invariant."""
    state = "state"
    q1 = gm.text.QuestionSpec(
        id="q1", text="q1", type=gm.text.QuestionType.NOUL
    )
    q2_alpha = gm.text.QuestionSpec(
        id="q2",
        text="q2",
        type=gm.text.QuestionType.CHOICE,
        options=["A", "B"],
    )
    q2_beta = gm.text.QuestionSpec(
        id="q2",
        text="completely different text with other tokens",
        type=gm.text.QuestionType.CHOICE,
        options=["A", "B"],
    )

    resp_a = self.sampler.evaluate_systemone(
        state=state, questions=[q1, q2_alpha], calibrate=False
    )
    resp_b = self.sampler.evaluate_systemone(
        state=state, questions=[q1, q2_beta], calibrate=False
    )

    q1_a = resp_a.decisions["q1"]
    q1_b = resp_b.decisions["q1"]

    self.assertAlmostEqual(
        q1_a.probabilities["true"], q1_b.probabilities["true"], delta=1e-9
    )
    self.assertAlmostEqual(
        q1_a.probabilities["false"], q1_b.probabilities["false"], delta=1e-9
    )
    self.assertAlmostEqual(q1_a.confidence, q1_b.confidence, delta=1e-9)
    self.assertAlmostEqual(q1_a.raw_entropy, q1_b.raw_entropy, delta=1e-9)

  def test_forward_pass_count_strictly_one(self):
    """Asserts forward pass count per evaluate call strictly equals 1."""
    state = "state"
    questions = [
        gm.text.QuestionSpec(
            id="q1", text="q1", type=gm.text.QuestionType.NOUL
        ),
        gm.text.QuestionSpec(
            id="q2",
            text="q2",
            type=gm.text.QuestionType.CHOICE,
            options=["A", "B"],
        ),
        gm.text.QuestionSpec(
            id="q3",
            text="q3",
            type=gm.text.QuestionType.SCORE,
            score_range=(1, 5),
        ),
    ]

    with mock.patch.object(
        self.model, "apply", wraps=self.model.apply
    ) as mock_apply:
      self.sampler.evaluate_systemone(
          state=state, questions=questions, calibrate=False
      )
      self.assertEqual(mock_apply.call_count, 1)

  def test_score_expected_value(self):
    """Asserts expected value computation Sum(k * P(k)) for SCORE primitive."""
    score_res = self.sampler.evaluate_score(
        state="state",
        question="q3",
        score_range=(1, 5),
        question_id="q3",
        calibrate=False,
    )
    self.assertIsInstance(score_res, gm.text.ScoreResult)
    self.assertIn(score_res.value, [1, 2, 3, 4, 5])
    expected_sum = sum(k * p for k, p in score_res.probabilities.items())
    self.assertAlmostEqual(score_res.expected_value, expected_sum, delta=1e-5)

  def test_null_calibration_debias(self):
    """Asserts null calibration applies z_cal = z_real - z_null."""
    cand_ids = (7, 10)  # "True", "False"
    # Artificially heavily bias toward "False" in null prior
    null_bias = jnp.array([-5.0, 5.0])
    self.sampler._null_cache[cand_ids] = null_bias

    res = self.sampler.evaluate_noul(
        state="state",
        question="q1",
        question_id="q1",
        calibrate=True,
    )
    # The null bias subtracted should boost "True" relative to raw
    raw_res = self.sampler.evaluate_noul(
        state="state",
        question="q1",
        question_id="q1",
        calibrate=False,
    )
    self.assertGreater(res.probabilities["true"], raw_res.probabilities["true"])

  def test_jit_compatibility(self):
    """Asserts core mathematical sub-routines compile cleanly under jax.jit."""
    # 1. JIT build_tree_attention_pack
    jitted_pack = jax.jit(
        gm.text.build_tree_attention_pack, static_argnums=(0, 1)
    )
    inp, pos, mask, terms = jitted_pack((1, 2), ((3, 4), (5, 6, 7)))
    self.assertEqual(inp.shape, (1, 7))
    self.assertEqual(mask.shape, (1, 1, 7, 7))

    # 2. JIT calibrate_and_score
    def _cal_fn(r, n):
      return gm.text.calibrate_and_score(r, n, temperature=1.0)

    jitted_cal = jax.jit(_cal_fn)
    probs, entropy, conf = jitted_cal(
        jnp.array([1.0, 2.0]), jnp.array([0.0, 0.0])
    )
    self.assertEqual(probs.shape, (2,))

  def test_cyclic_shifts(self):
    """Tests generation of cyclic permutation indices."""
    shifts_3 = gm.text.cyclic_shifts(3)
    self.assertEqual(shifts_3, [[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    shifts_4_max2 = gm.text.cyclic_shifts(4, max_permutations=2, spread=False)
    self.assertEqual(shifts_4_max2, [[0, 1, 2, 3], [1, 2, 3, 0]])

    shifts_4_spread = gm.text.cyclic_shifts(4, max_permutations=2, spread=True)
    self.assertEqual(shifts_4_spread, [[0, 1, 2, 3], [2, 3, 0, 1]])

  def test_marginalize_cyclic_distributions(self):
    """Tests logmean marginalization removes position bias."""
    perms = [[0, 1], [1, 0]]
    p_by_perm = np.array([[0.8, 0.2], [0.8, 0.2]])
    marg_probs = gm.text.marginalize_cyclic_distributions(
        p_by_perm, perms, combine="logmean"
    )
    self.assertAlmostEqual(marg_probs[0], 0.5, delta=1e-5)
    self.assertAlmostEqual(marg_probs[1], 0.5, delta=1e-5)

  def test_compute_order_flip_rate(self):
    """Tests computation of option order flip rate across permutations."""
    perms = [[0, 1], [1, 0]]
    p_flipped = np.array([[0.8, 0.2], [0.8, 0.2]])
    flip_rate = gm.text.compute_order_flip_rate(p_flipped, perms)
    self.assertEqual(flip_rate, 1.0)

    p_consistent = np.array([[0.8, 0.2], [0.2, 0.8]])
    flip_rate_consistent = gm.text.compute_order_flip_rate(p_consistent, perms)
    self.assertEqual(flip_rate_consistent, 0.0)

  def test_temperature_scaler_and_ece(self):
    """Tests post-hoc temperature fitting and ECE reduction."""
    np.random.seed(42)
    labels = np.random.randint(0, 3, size=60)
    overconfident_logits = np.random.randn(60, 3) * 6.0
    exp_z = np.exp(overconfident_logits)
    raw_probs = exp_z / exp_z.sum(axis=-1, keepdims=True)

    raw_ece = gm.text.compute_ece(raw_probs, labels)
    scaler = gm.text.TemperatureScaler.fit(raw_probs, labels)
    cal_probs = scaler.apply(raw_probs)
    cal_ece = gm.text.compute_ece(cal_probs, labels)

    self.assertGreater(scaler.temperature, 1.0)
    self.assertLess(cal_ece, raw_ece)

  def test_tree_attention_cyclic_marginalization_end_to_end(self):
    """Tests evaluate_choice with marginalize=True in a single pass."""
    with mock.patch.object(
        self.sampler, "_forward", wraps=self.sampler._forward
    ) as mock_forward:
      result = self.sampler.evaluate_choice(
          state="state",
          question="q2",
          options=["A", "B", "C"],
          marginalize=True,
      )
      self.assertEqual(mock_forward.call_count, 1)
      self.assertIsInstance(result, gm.text.ChoiceResult)
      self.assertIn(result.value, ["A", "B", "C"])
      self.assertIsNotNone(result.order_flip_rate)
      self.assertGreaterEqual(result.order_flip_rate, 0.0)
      self.assertLessEqual(result.order_flip_rate, 1.0)

  def test_choice_check_flip_rate(self):
    """Tests evaluate_choice with check_flip_rate=True."""
    with mock.patch.object(
        self.sampler, "_forward", wraps=self.sampler._forward
    ) as mock_forward:
      result = self.sampler.evaluate_choice(
          state="state",
          question="q2",
          options=["A", "B", "C"],
          check_flip_rate=True,
          marginalize=False,
      )
      self.assertEqual(mock_forward.call_count, 1)
      self.assertIsInstance(result, gm.text.ChoiceResult)
      self.assertIsNotNone(result.order_flip_rate)

  def test_pad_to_length(self):
    """Tests pad_to_length in build_tree_attention_pack and sampler."""
    inp, pos, mask, terms = gm.text.build_tree_attention_pack(
        state_ids=[1, 2],
        questions_ids=[[3, 4], [5]],
        pad_to_length=10,
    )
    self.assertEqual(inp.shape, (1, 10))
    self.assertEqual(pos.shape, (1, 10))
    self.assertEqual(mask.shape, (1, 1, 10, 10))
    self.assertEqual(len(terms), 2)
    # Mask padding region should be False
    self.assertFalse(bool(mask[0, 0, 5, 5]))

    # Exceeding pad_to_length raises ValueError
    with self.assertRaises(ValueError):
      gm.text.build_tree_attention_pack(
          state_ids=[1, 2],
          questions_ids=[[3, 4], [5]],
          pad_to_length=3,
      )

    # Sampler with pad_to_length returns valid decisions
    result = self.sampler.evaluate_choice(
        state="state",
        question="q2",
        options=["A", "B", "C"],
        pad_to_length=32,
    )
    self.assertIsInstance(result, gm.text.ChoiceResult)


if __name__ == "__main__":
  absltest.main()
