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

"""Thinking channel budget and context management utilities.

This module provides utilities for:
1. Managing thinking channel token budgets to prevent infinite loops
2. Optimizing long system prompts for better instruction following
3. Improving multi-turn conversation context retention

These utilities address known issues with Gemma 4 models:
- Infinite thinking loops in the <|channel>thought<channel|> block
- Degraded instruction following with system prompts >10k tokens
- Context loss in multi-turn conversations
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Sequence

import jax.numpy as jnp


# Recommended maximum system prompt length in tokens.
# Beyond this threshold, instruction following may degrade.
RECOMMENDED_MAX_SYSTEM_PROMPT_TOKENS = 8192

# Hard limit for system prompt tokens. Beyond this, behavior is unreliable.
HARD_MAX_SYSTEM_PROMPT_TOKENS = 16384

# Default thinking budget (in tokens) for Gemma 4 models.
DEFAULT_THINKING_BUDGET = 8192

# Minimum thinking budget. Below this, the model may not have enough tokens
# to complete its reasoning.
MIN_THINKING_BUDGET = 1024


@dataclasses.dataclass(frozen=True, kw_only=True)
class ThinkingBudgetConfig:
  """Configuration for thinking channel budget management.

  Attributes:
    max_thinking_tokens: Maximum number of tokens allowed inside the thinking
      channel block. When exhausted, the sampler forces an exit. Set to -1
      to disable (no limit).
    warn_on_budget_exhaustion: If True, emit a warning when the thinking
      budget is exhausted, indicating that the model's reasoning may be
      truncated.
    thinking_budget_safety_margin: Safety margin as a fraction of the thinking
      budget. When remaining budget drops below this fraction, start
      suppressing non-essential thinking tokens to encourage early exit.
      Default: 0.1 (10%).
  """

  max_thinking_tokens: int = DEFAULT_THINKING_BUDGET
  warn_on_budget_exhaustion: bool = True
  thinking_budget_safety_margin: float = 0.1

  def __post_init__(self):
    if self.max_thinking_tokens < -1:
      raise ValueError(
          f'max_thinking_tokens must be >= -1, got {self.max_thinking_tokens}'
      )
    if self.max_thinking_tokens >= 0 and self.max_thinking_tokens < MIN_THINKING_BUDGET:
      warnings.warn(
          f'max_thinking_tokens={self.max_thinking_tokens} is very low. '
          f'Minimum recommended is {MIN_THINKING_BUDGET}. The model may not '
          'have enough tokens to complete its reasoning.',
          UserWarning,
          stacklevel=2,
      )
    if not 0.0 <= self.thinking_budget_safety_margin <= 0.5:
      raise ValueError(
          'thinking_budget_safety_margin must be between 0.0 and 0.5, '
          f'got {self.thinking_budget_safety_margin}'
      )

  @property
  def effective_max_thinking_tokens(self) -> int:
    """Returns the effective max thinking tokens (-1 if disabled)."""
    return self.max_thinking_tokens

  def should_suppress_thinking(self, remaining: int) -> bool:
    """Returns True if thinking tokens should be suppressed (budget nearly exhausted)."""
    if self.max_thinking_tokens < 0:
      return False
    threshold = int(self.max_thinking_tokens * self.thinking_budget_safety_margin)
    return remaining <= threshold


@dataclasses.dataclass(frozen=True, kw_only=True)
class SystemPromptConfig:
  """Configuration for system prompt optimization.

  Attributes:
    max_system_prompt_tokens: Maximum recommended system prompt length in
      tokens. If exceeded, a warning is emitted and the prompt may be
      truncated during preprocessing.
    truncate_strategy: How to truncate long system prompts. Options:
      - 'warn': Emit a warning but do not truncate (default).
      - 'head_tail': Keep the first and last portions, drop the middle.
      - 'head_only': Keep only the beginning of the system prompt.
    system_prompt_priority_weight: Weight for system prompt tokens in the
      attention mask. Higher values give the model stronger attention to
      system instructions. Default: 1.0 (no change). Range: [0.5, 2.0].
  """

  max_system_prompt_tokens: int = RECOMMENDED_MAX_SYSTEM_PROMPT_TOKENS
  truncate_strategy: str = 'warn'
  system_prompt_priority_weight: float = 1.0

  def __post_init__(self):
    if self.truncate_strategy not in ('warn', 'head_tail', 'head_only'):
      raise ValueError(
          f'Unknown truncate_strategy: {self.truncate_strategy!r}. '
          "Must be 'warn', 'head_tail', or 'head_only'."
      )
    if not 0.5 <= self.system_prompt_priority_weight <= 2.0:
      raise ValueError(
          'system_prompt_priority_weight must be between 0.5 and 2.0, '
          f'got {self.system_prompt_priority_weight}'
      )

  def check_system_prompt_length(self, num_tokens: int) -> list[str]:
    """Check system prompt length and return warnings if needed.

    Args:
      num_tokens: Number of tokens in the system prompt.

    Returns:
      List of warning messages (empty if no issues).
    """
    warnings_list = []
    if num_tokens > HARD_MAX_SYSTEM_PROMPT_TOKENS:
      warnings_list.append(
          f'System prompt has {num_tokens} tokens, which exceeds the hard '
          f'limit of {HARD_MAX_SYSTEM_PROMPT_TOKENS}. Model behavior will be '
          'unreliable. Consider splitting the prompt into multiple turns or '
          'using a more concise system prompt.'
      )
    elif num_tokens > self.max_system_prompt_tokens:
      warnings_list.append(
          f'System prompt has {num_tokens} tokens, exceeding the recommended '
          f'maximum of {self.max_system_prompt_tokens}. Instruction following '
          'may degrade. Consider condensing the system prompt.'
      )
    return warnings_list


@dataclasses.dataclass(frozen=True, kw_only=True)
class MultiTurnConfig:
  """Configuration for multi-turn conversation context management.

  Attributes:
    context_refresh_threshold: After this many turns, suggest refreshing the
      context by re-injecting key system prompt instructions. Set to 0 to
      disable. Default: 10.
    max_context_tokens: Maximum total context length (system prompt + all
      turns). When exceeded, older turns are progressively summarized or
      dropped. Set to -1 to disable. Default: -1.
    context_priority_decay: Rate at which older turns lose attention priority.
      0.0 = no decay (all turns equal), 1.0 = linear decay. Default: 0.3.
  """

  context_refresh_threshold: int = 10
  max_context_tokens: int = -1
  context_priority_decay: float = 0.3

  def __post_init__(self):
    if self.context_refresh_threshold < 0:
      raise ValueError(
          'context_refresh_threshold must be >= 0, '
          f'got {self.context_refresh_threshold}'
      )
    if self.max_context_tokens < -1:
      raise ValueError(
          f'max_context_tokens must be >= -1, got {self.max_context_tokens}'
      )
    if not 0.0 <= self.context_priority_decay <= 1.0:
      raise ValueError(
          'context_priority_decay must be between 0.0 and 1.0, '
          f'got {self.context_priority_decay}'
      )


def truncate_system_prompt(
    tokens: list[int],
    max_tokens: int,
    strategy: str = 'head_tail',
) -> list[int]:
  """Truncate a system prompt to fit within a token budget.

  Args:
    tokens: The system prompt token IDs.
    max_tokens: Maximum number of tokens to keep.
    strategy: Truncation strategy ('head_tail' or 'head_only').

  Returns:
    Truncated token list.
  """
  if len(tokens) <= max_tokens:
    return tokens

  if strategy == 'head_only':
    return tokens[:max_tokens]
  elif strategy == 'head_tail':
    # Keep first 60% and last 40% of the budget
    head_size = int(max_tokens * 0.6)
    tail_size = max_tokens - head_size
    return tokens[:head_size] + tokens[-tail_size:]
  else:
    raise ValueError(f'Unknown truncation strategy: {strategy!r}')


def compute_turn_priority_weights(
    num_turns: int,
    system_prompt_length: int,
    decay: float = 0.3,
) -> list[float]:
  """Compute attention priority weights for multi-turn conversations.

  Later turns get higher priority, with the system prompt getting the
  highest priority. The decay parameter controls how quickly older turns
  lose priority.

  Args:
    num_turns: Number of conversation turns (excluding system prompt).
    system_prompt_length: Length of the system prompt in tokens.
    decay: Priority decay rate (0.0 = no decay, 1.0 = linear).

  Returns:
    List of priority weights, one per turn + system prompt.
  """
  weights = []

  # System prompt always gets the highest weight.
  weights.append(1.0 + decay)

  # Earlier turns get lower priority, later turns get higher priority.
  for i in range(num_turns):
    # Normalize turn index to [0, 1]
    if num_turns > 1:
      normalized_idx = i / (num_turns - 1)
    else:
      normalized_idx = 1.0
    # Apply decay: later turns get higher weight
    weight = 1.0 + decay * normalized_idx
    weights.append(weight)

  return weights


def should_refresh_context(
    turn_count: int,
    refresh_threshold: int,
) -> bool:
  """Check if context should be refreshed based on turn count.

  Args:
    turn_count: Current number of conversation turns.
    refresh_threshold: Number of turns after which to refresh.

  Returns:
    True if context should be refreshed.
  """
  if refresh_threshold <= 0:
    return False
  return turn_count > 0 and turn_count % refresh_threshold == 0

