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

"""Utility functions for vision encoders."""

from __future__ import annotations
from collections.abc import Sequence
from flax import linen as nn
import jax
from jax import numpy as jnp
from kauldron import typing
import numpy as np


def _posemb_sincos_2d(
    h: int,
    w: int,
    *,
    width: int,
    temperature: float = 10_000.0,
    dtype: jnp.dtype = jnp.float32,
) -> typing.Float["B M D"]:
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1.0 / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  block_id: int
  mlp_dim: int | None = None  # Defaults to 4x input dim
  dropout: float = 0.0
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    d = x.shape[-1]
    x = nn.Dense(features=self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(
        x
    )
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(
        features=d,
        dtype=self.dtype_mm,
        **inits,
    )(x)
    return x


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""

  block_id: int
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12
  buggy: bool = False

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    # TODO(lbeyer): condition on GAP(x)
    n, l, d = x.shape  # pylint: disable=unused-variable
    probe = self.param(
        "probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype
    )
    probe = jnp.tile(probe, [n, 1, 1])

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads, kernel_init=nn.initializers.xavier_uniform()
    )(probe, x)

    # TODO(lbeyer): dropout on head?
    y = nn.LayerNorm()(x)
    if self.buggy:
      x = y
    x = x + MlpBlock(self.block_id, mlp_dim=self.mlp_dim)(y)
    return x[:, 0]


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""

  block_id: int
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(
      self, x: jax.Array, deterministic: bool = True
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    y = nn.LayerNorm()(x)

    y = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
        dtype=self.dtype_mm,
    )(y, y)
    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = x + y

    y = nn.LayerNorm()(x)
    y = MlpBlock(
        block_id=self.block_id,
        mlp_dim=self.mlp_dim,
        dropout=self.dropout,
        dtype_mm=self.dtype_mm,
    )(y, deterministic)
    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = x + y
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    return x


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  depth: int
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  scan: bool = False
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
    if self.scan:
      block = nn.remat(
          Encoder1DBlock,
          prevent_cse=False,
          static_argnums=(2,),  # 0=self, 2=deterministic
          policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
      )
      x = nn.scan(
          block,
          variable_axes={"params": 0},
          split_rngs={"params": True, "dropout": True},
          in_axes=nn.broadcast,
          length=self.depth,
      )(
          block_id=0,
          name="encoderblock",
          dtype_mm=self.dtype_mm,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.dropout,
      )(
          x, deterministic
      )
    else:
      # Input Encoder
      for lyr in range(self.depth):
        block_cur = Encoder1DBlock(
            block_id=lyr,
            name=f"encoderblock_{lyr}",
            dtype_mm=self.dtype_mm,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        x = block_cur(x, deterministic)
    x: jax.Array = nn.LayerNorm(name="encoder_norm")(x)
    return x


class ViTModel(nn.Module):
  """ViT model.

  Attributes:
    compression_type: The compression type.
    width: The model dimension of the vision encoder.
    mlp_dim: The hidden dimension in the ffw layers.
    num_heads: The number of the heads.
    depth: The number of the layers.
    patch_size: The size to patchify images.
    posemb: The position embedding type.
    dropout: The dropout rate.
    scan: Whether to scan the layers (layer stacking).
    remat_policy: The remat policy.
    dtype_mm: The dtype to convert the input to.
    output_length: Number of soft tokens per image.
  """

  patch_size: Sequence[int] = (14, 14)
  width: int = 1152
  depth: int = 27
  mlp_dim: int | None = 4304  # Defaults to 4x input dim
  num_heads: int = 16
  posemb: str = "learn"  # Can also be "sincos2d"
  dropout: float = 0.0
  scan: bool = False
  # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"

  def _get_posemb(
      self,
      typ: str,
      *,
      seqshape: tuple[int, int],
      width: int,
      name: str,
      dtype: jnp.dtype = jnp.float32,
  ) -> typing.Float["B M D"]:
    """Returns the position embedding."""
    if typ == "learn":
      return self.param(
          name,
          nn.initializers.normal(stddev=1 / np.sqrt(width)),
          (1, np.prod(seqshape), width),
          dtype,
      )
    elif typ == "sincos2d":
      return _posemb_sincos_2d(*seqshape, width=width, dtype=dtype)
    else:
      raise ValueError(f"Unknown posemb type: {typ}")

  @nn.compact
  def __call__(
      self,
      image: typing.Float["B N P D"],
      *,
      train: bool = False,
  ):
    image = jnp.asarray(image, self.dtype_mm)

    # Patch extraction
    x = nn.Conv(
        self.width,
        self.patch_size,
        strides=self.patch_size,
        padding="VALID",
        name="embedding",
        dtype=self.dtype_mm,
    )(image)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Add posemb before adding extra token.
    x = x + self._get_posemb(
        self.posemb,
        seqshape=(h, w),
        width=c,
        name="pos_embedding",
        dtype=x.dtype,
    )

    n, l, c = x.shape  # pylint: disable=unused-variable
    x = nn.Dropout(rate=self.dropout)(x, not train)

    x = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        scan=self.scan,
        remat_policy=self.remat_policy,
        dtype_mm=self.dtype_mm,
        name="Transformer",
    )(x, deterministic=not train)

    return x
