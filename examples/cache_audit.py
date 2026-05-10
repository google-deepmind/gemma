"""Empirical KV-cache audit for the gemma sampler.

Standalone script that inspects the cache pytree allocated by the same
code path the sampler hits during prefill (gemma/gm/text/_prefill.py:295).
Calls `model.config.init_cache(...)` directly; that's the underlying
allocator. Skips the flax-module `nn.jit` wrapper because it requires a
bound module, but the allocation it produces is identical.

What it answers:
- Phase 2.1: load Gemma4_E2B (smallest) on whatever JAX backend is available.
- Phase 2.2: build the cache via `config.init_cache(cache_length=256)`.
- Phase 2.3: per-leaf shape, dtype, and sharding (or NONE if host-array).
- Phase 2.4: total cache bytes, broken down by attention type and tensor.
- Phase 2.5: re-run with cache_length=4096 and report the growth ratio.

Optional Hypothesis-B probe (--shard {none,fsdp,heads}):
- 'none'  : no sharding (default sampler behavior). Confirms replication.
- 'fsdp'  : shard along axis 0 (batch). What FSDPSharding-on-cache would do.
            With batch=1 this collapses to replication.
- 'heads' : shard k/v along axis 2 (num_kv_heads). The right sharding for
            batch=1 decode.

Usage on TPU node (no clone needed; uses the pip-installed gemma):

    python cache_audit.py
    python cache_audit.py --variant 31b --cache_lengths 256 4096
    python cache_audit.py --variant e4b --cache_mode local_window
    python cache_audit.py --variant 31b --cache_lengths 4096 --shard heads
"""

from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp


def _human_bytes(n: float) -> str:
  for unit in ("B", "KiB", "MiB", "GiB"):
    if n < 1024:
      return f"{n:.2f} {unit}"
    n /= 1024
  return f"{n:.2f} TiB"


def _fmt_sharding(arr) -> str:
  """Render an array's sharding compactly, or 'NONE' for host-only arrays."""
  if not isinstance(arr, jax.Array):
    return "NONE (host)"
  s = getattr(arr, "sharding", None)
  if s is None:
    return "NONE"
  cls = type(s).__name__
  spec = getattr(s, "spec", None)
  if spec is not None:
    return f"{cls}{tuple(spec)}"
  # SingleDeviceSharding / others.
  return cls


def _build_mesh(n_devices: int, *, tp_axis: int = None):
  """Build a 2D mesh; if tp_axis is None, all devices go to tensor."""
  from jax.sharding import Mesh
  import numpy as np
  if tp_axis is None or tp_axis >= n_devices:
    tp_axis = n_devices
  rep_axis = n_devices // tp_axis
  if rep_axis * tp_axis != n_devices:
    # Fall back to flat 1D 'tensor' mesh.
    devices = np.asarray(jax.devices()).reshape((n_devices,))
    return Mesh(devices, axis_names=("tensor",))
  devices = np.asarray(jax.devices()).reshape((rep_axis, tp_axis))
  return Mesh(devices, axis_names=("replicate", "tensor"))


def _gcd(a: int, b: int) -> int:
  while b:
    a, b = b, a % b
  return a


def _maybe_apply_sharding(cache, *, mode: str, mesh):
  """Apply a sharding to k/v leaves of the cache, or return as-is.

  - 'none'  : leave as-is (default sampler behavior).
  - 'fsdp'  : shard axis 0 (batch). At batch=1 this is impossible; we skip
              with a warning to mirror what kd.sharding.FSDPSharding would do.
  - 'heads' : shard axis 2 (num_kv_heads) for k/v. We pick the largest TP
              size that evenly divides every layer's head count and
              replicate over the remaining devices.
  """
  from jax.sharding import NamedSharding, PartitionSpec as P

  if mode == "none" or mesh is None:
    return cache

  if mode == "fsdp":
    sample = next(iter(cache.values()))["k"]
    if sample.shape[0] == 1:
      print("  [fsdp shard] batch=1 -> axis-0 sharding impossible; "
            "cache stays single-device. This is exactly the OOM pattern.")
      return cache
    # Else (rare for sampler), fall through to honoring the user's request.

  # Compute the largest tp size that evenly divides every layer's head count.
  head_counts = []
  for layer_name, layer_data in cache.items():
    head_counts.append(layer_data["k"].shape[2])
  n_dev = mesh.devices.size
  tp = n_dev
  for h in head_counts:
    tp = _gcd(tp, h)
  if tp <= 1:
    print(f"  [heads shard] no evenly-divisible TP size (head counts="
          f"{sorted(set(head_counts))}, devices={n_dev}); "
          "cache stays single-device.")
    return cache
  if tp != n_dev:
    print(f"  [heads shard] TP={tp} (heads divisible), replicate over "
          f"{n_dev // tp} chips. Per-chip memory ~= 1/{tp} of logical total.")

  # Rebuild a 2D mesh of (replicate, tensor) with these factors.
  from jax.sharding import Mesh
  import numpy as np
  rep = n_dev // tp
  devices_arr = np.asarray(jax.devices()).reshape((rep, tp)) if rep > 1 \
      else np.asarray(jax.devices()).reshape((tp,))
  if rep > 1:
    mesh_used = Mesh(devices_arr, axis_names=("replicate", "tensor"))
  else:
    mesh_used = Mesh(devices_arr, axis_names=("tensor",))

  def kv_spec_for(arr):
    rank = arr.ndim
    if mode == "fsdp":
      spec = P("tensor",) + (None,) * (rank - 1)
    elif mode == "heads":
      if rank == 4:
        spec = P(None, None, "tensor", None)
      else:
        spec = P(*(None,) * rank)
    else:
      raise ValueError(f"Unknown shard mode: {mode}")
    return NamedSharding(mesh_used, spec)

  out = {}
  for layer_name, layer_data in cache.items():
    new_layer = {}
    for tensor_name, arr in layer_data.items():
      try:
        new_layer[tensor_name] = jax.device_put(arr, kv_spec_for(arr))
      except Exception as e:  # pylint: disable=broad-except
        print(f"  [warn] could not shard {layer_name}/{tensor_name} "
              f"(shape={tuple(arr.shape)}): {e}; leaving as-is.")
        new_layer[tensor_name] = arr
    out[layer_name] = new_layer
  return out


def audit_cache(model, cache_length: int, *, batch_size: int, dtype,
                cache_mode, shard_mode: str, mesh) -> int:
  """Allocate a cache and dump per-leaf info. Returns total bytes."""
  print(f"\n=== cache_length={cache_length}, batch_size={batch_size}, "
        f"dtype={dtype.__name__ if hasattr(dtype, '__name__') else dtype}, "
        f"cache_mode={cache_mode.value}, shard={shard_mode} ===")

  # Allocate via the underlying allocator (same one the sampler hits).
  # See _prefill.py -> model.init_cache(...) -> config.init_cache.
  cache = model.config.init_cache(
      batch_size=batch_size,
      dtype=dtype,
      cache_length=cache_length,
      kv_cache_mode=cache_mode,
  )
  cache = _maybe_apply_sharding(cache, mode=shard_mode, mesh=mesh)

  total_bytes = 0
  by_kind = {}
  per_layer_bytes = []

  layer_names = sorted(cache.keys(), key=lambda s: int(s.split("_")[-1]))
  attn_types = list(model.config.attention_types)

  print(f"\n{'layer':>9}  {'attn':>6}  {'tensor':>10}  {'shape':>32}  "
        f"{'dtype':>10}  {'bytes':>12}  sharding")
  print("-" * 130)

  sample_layer_idxs = {0, 1, len(layer_names) // 2, len(layer_names) - 1}
  for i, layer_name in enumerate(layer_names):
    layer_data = cache[layer_name]
    layer_total = 0
    attn = attn_types[i].name[:6]
    for tensor_name, arr in layer_data.items():
      shape = tuple(arr.shape)
      dt = arr.dtype
      nbytes = int(arr.size) * arr.itemsize
      total_bytes += nbytes
      layer_total += nbytes
      cnt, b = by_kind.get(tensor_name, (0, 0))
      by_kind[tensor_name] = (cnt + 1, b + nbytes)
      if i in sample_layer_idxs:
        print(f"{layer_name:>9}  {attn:>6}  {tensor_name:>10}  "
              f"{str(shape):>32}  {str(dt):>10}  {_human_bytes(nbytes):>12}  "
              f"{_fmt_sharding(arr)}")
    per_layer_bytes.append((layer_name, attn, layer_total))

  print(f"\n[{len(layer_names)} layers total; only sample layers shown above]")

  print("\nBy tensor kind:")
  for kind, (cnt, b) in sorted(by_kind.items()):
    print(f"  {kind:>10}: {cnt:>3} arrays, {_human_bytes(b):>12}")

  local_bytes = sum(b for _, a, b in per_layer_bytes if a == "LOCAL_")
  global_bytes = sum(b for _, a, b in per_layer_bytes if a == "GLOBAL")
  if local_bytes or global_bytes:
    n_local = sum(1 for _, a, _ in per_layer_bytes if a == "LOCAL_")
    n_global = sum(1 for _, a, _ in per_layer_bytes if a == "GLOBAL")
    print("\nBy attention type:")
    print(f"  LOCAL_SLIDING ({n_local:>3} layers): {_human_bytes(local_bytes)}")
    print(
        f"  GLOBAL        ({n_global:>3} layers): "
        f"{_human_bytes(global_bytes)}"
    )

  print(f"\nTOTAL cache size (logical, summed over all layers): "
        f"{_human_bytes(total_bytes)} ({total_bytes:,} bytes)")
  if mesh is not None and shard_mode == "heads":
    # Determine the actual TP factor used (gcd of all head counts and ndev).
    head_counts = [cache[ln]["k"].shape[2] for ln in layer_names]
    n_dev = mesh.devices.size
    tp = n_dev
    for h in head_counts:
      tp = _gcd(tp, h)
    rep = n_dev // max(1, tp)
    if tp <= 1:
      print("  TP=1 (no sharding possible); cache stays single-device.")
    else:
      per_chip = total_bytes / tp
      print(f"  TP={tp}, replicate={rep}: each chip holds 1/{tp} of the "
            f"cache => ~{_human_bytes(per_chip)} per chip.")
      print(f"  HBM consumed across whole {n_dev}-chip slice: "
            f"~{_human_bytes(per_chip * n_dev)} "
            f"(per_chip x {n_dev} chips, since "
            f"{rep} group(s) replicate the same shards).")
  elif shard_mode == "none":
    print("  Cache is unconstrained; `jnp.zeros` outside any jit lands on "
          "JAX's default device. Expected: SingleDeviceSharding on chip 0. "
          "Inside the real sampler's `nn.jit` it may instead replicate "
          "across the active mesh, but in EITHER case it is not sharded "
          "across chips.")

  first_k = cache[layer_names[0]]["k"]
  if isinstance(first_k, jax.Array) and jax.local_device_count() > 1:
    print("\nSharding visualization for cache[layer_0]['k']:")
    try:
      jax.debug.visualize_array_sharding(first_k)
    except Exception as e:  # pylint: disable=broad-except
      print(f"  (visualize_array_sharding failed: {e})")

  return total_bytes


def build_model(variant: str):
  """Construct a Gemma 4 flax module without loading params."""
  from gemma import gm  # pylint: disable=g-import-not-at-top
  cls = {
      "e2b": gm.nn.Gemma4_E2B,
      "e4b": gm.nn.Gemma4_E4B,
      "31b": gm.nn.Gemma4_31B,
      "26b_a4b": gm.nn.Gemma4_26B_A4B,
  }[variant.lower()]
  return cls()


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--variant", default="e2b",
                      choices=("e2b", "e4b", "31b", "26b_a4b"))
  parser.add_argument("--cache_lengths", type=int, nargs="+",
                      default=[256, 4096])
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--dtype", default="bfloat16",
                      choices=("bfloat16", "float16", "float32"))
  parser.add_argument("--cache_mode", default="legacy",
                      choices=("legacy", "local_window"),
                      help="KV cache allocation mode to audit.")
  parser.add_argument("--shard", default="none",
                      choices=("none", "fsdp", "heads"),
                      help="Sharding strategy for the cache "
                           "(see module docstring).")
  args = parser.parse_args(argv)

  dtype = {"bfloat16": jnp.bfloat16, "float16": jnp.float16,
           "float32": jnp.float32}[args.dtype]
  from gemma.gm.utils import (  # pylint: disable=g-import-not-at-top
      _cache_helper,
  )
  cache_mode = _cache_helper.KVCacheMode(args.cache_mode)

  print(f"JAX version: {jax.__version__}")
  print(f"JAX devices: {jax.devices()}")
  n_dev = jax.local_device_count()
  mesh = _build_mesh(n_dev) if n_dev > 1 else None
  if mesh is not None:
    print(f"Mesh: {mesh}")
  elif args.shard != "none":
    print(f"WARNING: only {n_dev} device(s) visible; --shard={args.shard} "
          "will fall back to single-device placement.")

  print(f"Building Gemma4_{args.variant.upper()} (no params load)...")
  model = build_model(args.variant)
  cfg = model.config
  print(f"  num_layers          = {cfg.num_layers}")
  print(f"  num_kv_heads        = {cfg.num_kv_heads}")
  print(f"  num_global_kv_heads = {cfg.num_global_kv_heads}")
  print(f"  head_dim            = {cfg.head_dim}")
  print(f"  global_key_size     = {cfg.global_key_size}")
  print(f"  sliding_window_size = {cfg.sliding_window_size}")
  n_local = sum(1 for a in cfg.attention_types if a.name == "LOCAL_SLIDING")
  n_global = sum(1 for a in cfg.attention_types if a.name == "GLOBAL")
  print(f"  local layers        = {n_local}")
  print(f"  global layers       = {n_global}")

  results = {}
  for L in args.cache_lengths:
    results[L] = audit_cache(
        model, cache_length=L, batch_size=args.batch_size,
        dtype=dtype, cache_mode=cache_mode, shard_mode=args.shard, mesh=mesh,
    )

  if len(results) >= 2:
    Ls = sorted(results)
    base = results[Ls[0]]
    print(f"\n{'=' * 60}")
    print("Cache size scaling with cache_length:")
    for L in Ls:
      ratio = results[L] / base
      print(f"  cache_length={L:>6}: {_human_bytes(results[L]):>12}  "
            f"(x{ratio:.2f} of cache_length={Ls[0]})")
    expected = Ls[-1] / Ls[0]
    actual = results[Ls[-1]] / results[Ls[0]]
    print(f"\nExpected ratio (linear in cache_length): x{expected:.2f}")
    print(f"Actual ratio                            : x{actual:.2f}")
    if cache_mode == _cache_helper.KVCacheMode.LOCAL_WINDOW:
      print("=> LOCAL_WINDOW mode should scale sub-linearly once local layers "
            "hit the sliding-window cap; global layers still scale linearly.")
    elif abs(actual - expected) / expected < 0.05:
      print("=> Cache scales LINEARLY with cache_length on every layer.")
      print("   Confirms eager pre-allocation across all layers,")
      print("   including LOCAL_SLIDING layers that only need the window.")
    else:
      print("=> Cache scales SUB-LINEARLY; some layers are size-capped.")

  return 0


if __name__ == "__main__":
  sys.exit(main())
