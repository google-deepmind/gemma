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

"""Utils to parse `einsum` strings."""

from collections.abc import Sequence
import string

_Shape = Sequence[int]


def get_lora_einsum_str_and_shapes(
    *,
    einsum_str: str,
    weights_shape: _Shape,
    rank: int,
) -> tuple[str, _Shape, _Shape]:
  """Extract the LoRA decomposition from the original einsum parameters.

  This function reqrites a einsum string `inputs,weights->outputs` into
  `inputs,a,b->outputs`.

  Args:
    einsum_str: The original einsum string (e.g. `'BTNH,NHD->BTD'`)
    weights_shape: The shape of the original einsum weights (e.g. `(N, H, D)`)
    rank: The rank of the LoRA decomposition.

  Returns:
    A tuple of (lora_einsum_str, a_shape, b_shape), e.g.
    `('BTNH,NHr,rD->BTD', (N, H, r), (r, D))`
  """

  # e.g. split `'BTNH,NHD->BTD'` into `('BTNH', 'NHD', 'BTD')`.
  inputs, weights, outputs = _split_einsum_str(einsum_str)

  # Extract the inputs dimensions which will be reduced and the outputs
  # dimensions.
  # Example: BTNH,NHD->BTD
  # * in_dims: NH
  # * out_dims: D
  in_dims = set(inputs) & set(weights) - set(outputs)  # Reduced dims
  out_dims = set(outputs) & set(weights) - set(inputs)
  untouched_dims = set(inputs) & set(weights) & set(outputs)  # Batch dims
  all_dims = set(inputs) | set(weights) | set(outputs)

  # Set is not deterministic, so restore the order from einsum_str.
  in_dims = tuple(c for c in weights if c in in_dims | untouched_dims)
  out_dims = tuple(c for c in weights if c in out_dims)

  # Add the `rank` dimension:
  # e.g. `NHD` is split into `NHr,rD`
  rank_dim = _find_unused_letter(all_dims)  # Choose a new letter for the rank
  a_str = ''.join(in_dims + (rank_dim,))
  b_str = ''.join((rank_dim,) + out_dims)

  lora_einsum_str = f'{inputs},{a_str},{b_str}->{outputs}'

  # This assume there's no elipsis in the weights.
  weights_str_to_dim = dict(zip(weights, weights_shape))
  weights_str_to_dim[rank_dim] = rank
  a_shape = tuple(weights_str_to_dim[c] for c in a_str)
  b_shape = tuple(weights_str_to_dim[c] for c in b_str)

  return (lora_einsum_str, a_shape, b_shape)


def _split_einsum_str(einsum_str: str) -> tuple[str, str, str]:
  """Splits an einsum string into its components."""

  # TODO(epot): Check length
  def _check_len2(x):
    if len(x) != 2:
      raise ValueError(
          f'`einsum_str` should be `inputs,weights->outputs` Got: {einsum_str}'
      )
    return x

  einsum_str = einsum_str.replace(' ', '')  # Strip whitespace
  inputs, outputs = _check_len2(einsum_str.split('->'))
  inputs, weights = _check_len2(inputs.split(','))
  if '...' in weights:
    raise ValueError(
        'Weights in `einsum_str` should not contain ellipsis. Got:'
        f' {weights} for {einsum_str}'
    )
  return (inputs, weights, outputs)


def _find_unused_letter(chars) -> str:
  """Returns an unused letter from a set of characters."""
  # einsum is case sensitive, so use both upper and lower case letters.
  if 'r' not in chars:  # If `r` isn't used, use it.
    return 'r'
  # Otherwise, find the first unused letter.
  return sorted(set(string.ascii_letters) - set(chars))[0]
