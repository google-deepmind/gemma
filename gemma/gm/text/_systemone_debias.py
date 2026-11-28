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

"""Calibration, temperature scaling, and option debiasing algorithms."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def spread_order(k: int) -> List[int]:
  """Computes van der Corput spread order of cyclic shifts for option debiasing.

  Consecutive shifts move every option by only 1 position. Spread shifts (e.g.
  0, k/2, k/4, 3k/4, ...) place each option in well-separated positions across
  the list, so a small subset of shifts approximates the full marginal.

  Args:
    k: Total number of options.

  Returns:
    List of shift offsets in spread order.
  """
  order, seen = [0], {0}
  d = 2
  while len(order) < k and d <= 4 * k:
    for num in range(1, d, 2):
      s = int(k * num / d) % k
      if s not in seen:
        seen.add(s)
        order.append(s)
    d *= 2
  order.extend(s for s in range(k) if s not in seen)
  return order[:k]


def cyclic_shifts(
    k: int,
    max_permutations: Optional[int] = None,
    spread: bool = True,
) -> List[List[int]]:
  """Generates cyclic permutation indices for option debiasing.

  perm[j] is the original option index displayed at position j.

  Args:
    k: Total number of options.
    max_permutations: Optional upper bound on shifts to generate.
    spread: If True, uses van der Corput spread order (e.g. 0, k/2, k/4...) so
      early subsets sample maximally distant positions.

  Returns:
    List of permutation lists, each of length k.
  """
  n = k if max_permutations is None else max(1, min(k, max_permutations))
  shifts = spread_order(k)[:n] if spread else list(range(n))
  return [[(j + s) % k for j in range(k)] for s in shifts]


def marginalize_cyclic_distributions(
    p_by_perm: Sequence[Sequence[float]] | np.ndarray,
    perms: Sequence[Sequence[int]],
    combine: str = "logmean",
) -> np.ndarray:
  """Combines distributions across cyclic permutations into option space.

  Args:
    p_by_perm: Shape [P, K] distributions indexed by displayed position.
    perms: Sequence of P permutations mapping position to original option index.
    combine: Aggregation method ('logmean' geometric mean or 'mean' arithmetic).

  Returns:
    Normalized [K] probability distribution indexed by original option order.
  """
  p_arr = np.asarray(p_by_perm, dtype=np.float64)
  p_count, k = p_arr.shape
  per_option = np.zeros((p_count, k), dtype=np.float64)
  for s, perm in enumerate(perms):
    for j, orig_idx in enumerate(perm):
      per_option[s, orig_idx] = p_arr[s, j]

  if combine == "logmean":
    eps = 1e-12
    z = np.log(np.clip(per_option, eps, None)).mean(axis=0)
    z = z - np.max(z)
    out = np.exp(z)
  elif combine == "mean":
    out = per_option.mean(axis=0)
  else:
    raise ValueError(f"combine must be 'logmean' or 'mean', got {combine!r}")

  total = np.sum(out)
  if total <= 0:
    return np.ones(k, dtype=np.float64) / k
  return out / total


def compute_order_flip_rate(
    p_by_perm: Sequence[Sequence[float]] | np.ndarray,
    perms: Sequence[Sequence[int]],
) -> float:
  """Fraction of permutations whose argmax disagrees with shift 0.

  Args:
    p_by_perm: Shape [P, K] distributions indexed by displayed position.
    perms: Sequence of P permutations mapping position to original option index.

  Returns:
    Float flip rate in [0.0, 1.0]. 0.0 indicates complete order invariance.
  """
  p_arr = np.asarray(p_by_perm, dtype=np.float64)
  if len(perms) <= 1:
    return 0.0
  winners = [perms[s][int(np.argmax(p_arr[s]))] for s in range(len(perms))]
  return float(np.mean([w != winners[0] for w in winners[1:]]))


def calibrate_and_score(
    real_logits: jnp.ndarray,
    null_logits: Optional[jnp.ndarray] = None,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, float, float]:
  """Computes calibrated probabilities, Shannon entropy, and confidence.

  Args:
    real_logits: Unnormalized logits for candidate tokens.
    null_logits: Optional baseline logits evaluated against a null state.
    temperature: Softmax readout temperature.

  Returns:
    probs: Probability distribution over candidates.
    entropy: Shannon entropy H(P) in bits.
    confidence: Normalized confidence in [0.0, 1.0].
  """
  # 1. Null Context Debias
  if null_logits is not None:
    calibrated_logits = real_logits - null_logits
  else:
    calibrated_logits = real_logits

  # 2. Subspace Softmax with Temperature
  probs = jax.nn.softmax(calibrated_logits / temperature, axis=-1)

  # 3. Shannon Entropy: H(P) = -Sum(p * log2(p))
  clipped_probs = jnp.clip(probs, 1e-12, 1.0)
  entropy = -jnp.sum(probs * jnp.log2(clipped_probs))

  # 4. Normalized Jev Confidence: 1.0 - (H(P) / log2(M))
  num_options = probs.shape[-1]
  if num_options > 1:
    max_entropy = math.log2(num_options)
    confidence = jnp.clip(1.0 - (entropy / max_entropy), 0.0, 1.0)
  else:
    confidence = 1.0

  if isinstance(entropy, jax.core.Tracer):
    return probs, entropy, confidence
  return probs, float(entropy), float(confidence)


@dataclasses.dataclass(frozen=True)
class TemperatureScaler:
  """Post-hoc temperature scaling (Guo et al., ICML 2017) to minimize ECE."""

  temperature: float = 1.0

  def apply(
      self, logits_or_probs: Sequence[Sequence[float]] | np.ndarray
  ) -> np.ndarray:
    """Scales logits or probabilities by temperature to calibrated probs."""
    arr = np.asarray(logits_or_probs, dtype=np.float64)
    eps = 1e-12
    if np.all(arr >= 0.0) and np.allclose(arr.sum(axis=-1), 1.0, atol=1e-3):
      logits = np.log(np.clip(arr, eps, None))
    else:
      logits = arr
    temp = max(self.temperature, 1e-6)
    scaled_logits = logits / temp
    scaled_logits = scaled_logits - np.max(
        scaled_logits, axis=-1, keepdims=True
    )
    exp_logits = np.exp(scaled_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

  @classmethod
  def fit(
      cls,
      logits_or_probs: Sequence[Sequence[float]] | np.ndarray,
      labels: Sequence[int] | np.ndarray,
      log_t_range: Tuple[float, float] = (-3.0, 3.0),
      iters: int = 60,
  ) -> "TemperatureScaler":
    """Fits scalar temperature minimizing NLL via golden-section search."""
    arr = np.asarray(logits_or_probs, dtype=np.float64)
    eps = 1e-12
    if np.all(arr >= 0.0) and np.allclose(arr.sum(axis=-1), 1.0, atol=1e-3):
      logits = np.log(np.clip(arr, eps, None))
    else:
      logits = arr

    labels_arr = np.asarray(labels, dtype=np.int32)
    if logits.ndim != 2:
      raise ValueError(
          f"logits_or_probs must be 2D [N, K], got shape {logits.shape}"
      )
    if len(labels_arr) != len(logits):
      raise ValueError(
          f"Length mismatch: {len(logits)} samples vs {len(labels_arr)} labels"
      )

    def _nll(scaled_z: np.ndarray) -> float:
      z_max = np.max(scaled_z, axis=-1, keepdims=True)
      log_sum_exp = z_max + np.log(
          np.sum(np.exp(scaled_z - z_max), axis=-1, keepdims=True)
      )
      log_p = scaled_z - log_sum_exp
      return float(-log_p[np.arange(len(labels_arr)), labels_arr].mean())

    lo, hi = log_t_range
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = _nll(logits / math.exp(c))
    fd = _nll(logits / math.exp(d))

    for _ in range(iters):
      if fc < fd:
        b, d, fd = d, c, fc
        c = b - phi * (b - a)
        fc = _nll(logits / math.exp(c))
      else:
        a, c, fc = c, d, fd
        d = a + phi * (b - a)
        fd = _nll(logits / math.exp(d))

    opt_temp = float(math.exp((a + b) / 2.0))
    return cls(temperature=opt_temp)


def fit_temperature(
    logits_or_probs: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    log_t_range: Tuple[float, float] = (-3.0, 3.0),
    iters: int = 60,
) -> float:
  """Fits a scalar temperature minimizing NLL on a validation set."""
  scaler = TemperatureScaler.fit(
      logits_or_probs, labels, log_t_range=log_t_range, iters=iters
  )
  return scaler.temperature


def compute_ece(
    probs: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    num_bins: int = 10,
) -> float:
  """Computes Expected Calibration Error (ECE) for probability distributions."""
  probs_arr = np.asarray(probs, dtype=np.float64)
  labels_arr = np.asarray(labels, dtype=np.int32)
  if probs_arr.ndim != 2:
    raise ValueError(f"probs must be 2D [N, K], got shape {probs_arr.shape}")
  if len(labels_arr) != len(probs_arr):
    raise ValueError(
        f"Length mismatch: {len(probs_arr)} samples vs {len(labels_arr)} labels"
    )

  confs = np.max(probs_arr, axis=-1)
  preds = np.argmax(probs_arr, axis=-1)
  correct = preds == labels_arr
  bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
  ece = 0.0
  n = len(labels_arr)

  for i in range(num_bins):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    if i == num_bins - 1:
      mask = (confs >= lo) & (confs <= hi)
    else:
      mask = (confs >= lo) & (confs < hi)
    if np.any(mask):
      bin_acc = np.mean(correct[mask])
      bin_conf = np.mean(confs[mask])
      bin_weight = np.sum(mask) / n
      ece += bin_weight * np.abs(bin_acc - bin_conf)

  return float(ece)
