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

"""Vision transformer implementation for Gemma4."""

from flax import linen as nn
from gemma.gm.nn.gemma4.vision import _modules
from kauldron.ktyping import Bool, Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class VisionBlock(nn.Module):
  """Wraps the core transformer block to be compatible with nn.scan."""

  d_model: int = 1152
  ffw_hidden: int = 4304
  num_heads: int = 16
  num_kv_heads: int = 16
  key_size: int = 128
  use_clipped_linears: bool = False

  def setup(self):
    self.block = _modules.Block(
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        embed_dim=self.d_model,
        head_dim=self.key_size,
        hidden_dim=self.ffw_hidden,
        rope_base_frequency=100,
        rope_scale_factor=1.0,
        use_qk_norm=True,
        use_clipped_linears=self.use_clipped_linears,
    )

  @typechecked
  def __call__(
      self,
      inputs: Float['B L D'],
      attn_mask: Bool['B #L l'] | None,
      positions: Int['B L'] | Int['B L 2'],
  ) -> tuple[Float['B L D'], None]:
    """Calls the block."""
    outputs = self.block(
        inputs=inputs, attn_mask=attn_mask, positions=positions
    )
    return outputs, None


class VisionTransformer(nn.Module):
  """The vision transformer layer.

  Attributes:
    d_model: The model dimension.
    ffw_hidden: The hidden dimension in the ffw layer.
    num_heads: The number of heads.
    num_layers: The number of the layers.
  """

  d_model: int
  ffw_hidden: int
  num_heads: int
  num_layers: int
  use_clipped_linears: bool = False

  def setup(self):
    scan_init_fn = nn.scan(
        VisionBlock,
        length=self.num_layers,
        split_rngs={'params': True, 'dropout': True},
        # in_axes=(nn.broadcast, nn.broadcast, nn.broadcast),
        in_axes=(nn.broadcast, nn.broadcast),
        out_axes=0,
        variable_axes={
            'params': 0,
            'metrics': 0,
            'aux_losses': 0,
            'moe_aux': 0,
        },
        metadata_params={
            'mesh_axis': None,
            'axis_type': 'stacked',
            'loss_reduction': 'mean',
        },
        unroll=1,
    )
    self.stacked_layers = scan_init_fn(
        d_model=self.d_model,
        ffw_hidden=self.ffw_hidden,
        num_heads=self.num_heads,
        num_kv_heads=self.num_heads,
        key_size=self.d_model // self.num_heads,
        use_clipped_linears=self.use_clipped_linears,
    )

  @typechecked
  def __call__(
      self,
      inputs: Float['B L D'],
      input_mask: Bool['B L'],
      positions_xy: Int['B L 2'] | None = None,
  ) -> Float['B L D']:
    assert positions_xy is not None
    attn_mask = input_mask[:, :, None] * input_mask[:, None, :]
    outputs = self.stacked_layers(
        inputs, attn_mask, positions_xy
    )[0]
    return outputs
