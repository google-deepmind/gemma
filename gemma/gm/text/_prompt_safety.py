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

"""Validation for structured dialog prompts."""

import dialog

_GEMMA4_ROLE_TOKENS = (
    dialog.Tags.TURN.open,
    dialog.Tags.TURN.close,
    dialog.Tags.CHANNEL.open,
    dialog.Tags.CHANNEL.close,
)


def validate_conversation(
    conversation: dialog.Conversation,
    *,
    format: dialog.Format,  # pylint: disable=redefined-builtin
) -> None:
  """Rejects role-control tokens embedded in structured message content."""
  role_tokens = tuple(
      dict.fromkeys((
          *_GEMMA4_ROLE_TOKENS,
          *(format.from_gemma4(t) for t in _GEMMA4_ROLE_TOKENS),
      ))
  )
  for turn in conversation:
    for chunk in turn:
      _validate_chunk(chunk, role_tokens)


def _validate_chunk(chunk: dialog.Chunk, role_tokens: tuple[str, ...]) -> None:
  if isinstance(chunk, dialog.ControlToken):
    return
  if isinstance(chunk, dialog.Thought):
    for child in chunk:
      _validate_chunk(child, role_tokens)
    return

  text = chunk.text if isinstance(chunk, dialog.Text) else chunk.as_text()
  token = next((token for token in role_tokens if token in text), None)
  if token is not None:
    raise ValueError(
        'dialog.Conversation message content contains reserved role token '
        f'{token!r}. Use structured dialog objects for control tokens, or pass '
        'an intentionally preformatted raw string to Sampler.'
    )
