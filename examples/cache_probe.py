"""Runtime HBM probe for the gemma sampler.

Measures per-device HBM usage before and after a real `ChatSampler.chat()`
call, so you can see which chip(s) absorb the cache. No library edits
needed; this is pure user-side instrumentation.

What it does:
1. Snapshots `bytes_in_use` per TPU chip via `device.memory_stats()`.
2. Loads E2B (default) or another Gemma 4 variant + params.
3. Builds a `gm.text.ChatSampler` exactly like a normal user would.
4. Snapshots HBM. Runs `sampler.chat(...)`. Snapshots HBM again.
5. Reports per-chip delta. The chip(s) that absorbed the cache will spike
   while the others stay flat; this is the per-chip evidence for
   Hypothesis B.

Usage on TPU:

    python cache_probe.py                                  # E2B, 4096 cache
    python cache_probe.py --variant e4b --cache_length 4096
    python cache_probe.py --variant e4b --cache_length 4096 \
        --cache_mode local_window
    python cache_probe.py --variant e4b --cache_length 16384

Notes:
- This loads real params, so it requires the kaggle/gcs checkpoint to be
  accessible. If you've been running ChatSampler successfully on this
  node, this will work the same way (same `gm.ckpts.LoadCheckpoint` /
  `gm.ckpts.CheckpointPath` plumbing).
- We don't pass `sharding=` to the sampler; that mirrors the documented
  pattern from colabs/sharding.ipynb. Set `--shard params_fsdp` to apply
  FSDP sharding to params before sampling (still no cache sharding).
"""

from __future__ import annotations

import argparse
import sys
import time

import jax
import jax.numpy as jnp


def _human(b: float) -> str:
  for u in ("B", "KiB", "MiB", "GiB"):
    if abs(b) < 1024:
      return f"{b:+.2f} {u}"
    b /= 1024
  return f"{b:+.2f} TiB"


def _abs(b: float) -> str:
  for u in ("B", "KiB", "MiB", "GiB"):
    if b < 1024:
      return f"{b:.2f} {u}"
    b /= 1024
  return f"{b:.2f} TiB"


def hbm_snapshot() -> dict[int, dict[str, int]]:
  """Per-device HBM stats. Returns {device_id: {bytes_in_use, peak, limit}}."""
  out = {}
  for d in jax.devices():
    if not hasattr(d, "memory_stats"):
      continue
    try:
      stats = d.memory_stats() or {}
    except Exception:  # pylint: disable=broad-except
      stats = {}
    out[d.id] = {
        "bytes_in_use": int(stats.get("bytes_in_use", 0)),
        "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
        "bytes_limit": int(stats.get("bytes_limit", 0)),
    }
  return out


def print_snapshot(title: str, snap: dict[int, dict[str, int]]) -> None:
  print(f"\n--- {title} ---")
  print(f"  {'chip':>4}  {'in_use':>14}  {'peak':>14}  {'limit':>14}")
  for d_id in sorted(snap):
    s = snap[d_id]
    print(f"  {d_id:>4}  {_abs(s['bytes_in_use']):>14}  "
          f"{_abs(s['peak_bytes_in_use']):>14}  "
          f"{_abs(s['bytes_limit']):>14}")


def print_delta(pre, post, title="Per-chip delta from chat()") -> None:
  print(f"\n--- {title} ---")
  print(f"  {'chip':>4}  {'pre':>14}  {'post':>14}  {'delta':>14}")
  total = 0
  for d_id in sorted(pre):
    pre_b = pre[d_id]["bytes_in_use"]
    post_b = post[d_id]["bytes_in_use"]
    delta = post_b - pre_b
    total += delta
    print(f"  {d_id:>4}  {_abs(pre_b):>14}  {_abs(post_b):>14}  "
          f"{_human(delta):>14}")
  print(f"  total delta across all chips: {_human(total)}")


def build_sampler(
    variant: str, *, cache_length: int, cache_mode, params_fsdp: bool
):
  from gemma import gm  # pylint: disable=g-import-not-at-top

  variants = {
      "e2b": (gm.nn.Gemma4_E2B, "GEMMA4_E2B_IT"),
      "e4b": (gm.nn.Gemma4_E4B, "GEMMA4_E4B_IT"),
      "31b": (gm.nn.Gemma4_31B, "GEMMA4_31B_IT"),
  }
  cls, ckpt_name = variants[variant.lower()]
  print(f"Building model {cls.__name__}...")
  model = cls()

  print(f"Loading checkpoint {ckpt_name} "
        f"(params_fsdp={params_fsdp})...")
  ckpt_path = getattr(gm.ckpts.CheckpointPath, ckpt_name)
  if params_fsdp:
    # Without FSDP, large checkpoints (E4B / 31B) try to land on a single
    # chip and OOM during deserialization, even before any sampler call.
    # FSDP-shards the params across the slice during load_params.
    from kauldron import kd  # pylint: disable=g-import-not-at-top
    sharding = kd.sharding.FSDPSharding()
    params = gm.ckpts.load_params(ckpt_path, sharding=sharding)
  else:
    params = gm.ckpts.load_params(ckpt_path)

  print(f"Constructing ChatSampler(cache_length={cache_length}, "
        f"cache_mode={cache_mode.value})...")
  sampler = gm.text.ChatSampler(
      model=model,
      params=params,
      cache_length=cache_length,
      kv_cache_mode=cache_mode,
  )
  return sampler


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--variant", default="e2b",
                      choices=("e2b", "e4b", "31b"))
  parser.add_argument("--cache_length", type=int, default=4096)
  parser.add_argument("--cache_mode", default="legacy",
                      choices=("legacy", "local_window"),
                      help="KV cache allocation mode to probe.")
  parser.add_argument("--prompt", default="Tell me a short fun fact.")
  parser.add_argument("--max_new_tokens", type=int, default=64,
                      help="Cap output length so probe is fast.")
  parser.add_argument("--params_fsdp", action=argparse.BooleanOptionalAction,
                      default=True,
                      help="Apply FSDP sharding to params during load. "
                           "ON by default (mirrors production deployment "
                           "and prevents single-chip OOM during checkpoint "
                           "load on E4B+). Use --no-params_fsdp to disable.")
  args = parser.parse_args(argv)

  print(f"JAX version: {jax.__version__}")
  print(f"JAX devices: {jax.devices()}")
  from gemma.gm.utils import (  # pylint: disable=g-import-not-at-top
      _cache_helper,
  )
  cache_mode = _cache_helper.KVCacheMode(args.cache_mode)

  print_snapshot("HBM at startup (no params loaded yet)", hbm_snapshot())

  sampler = build_sampler(
      args.variant,
      cache_length=args.cache_length,
      cache_mode=cache_mode,
      params_fsdp=args.params_fsdp,
  )

  pre = hbm_snapshot()
  print_snapshot("HBM after sampler construction (params loaded)", pre)

  print(f"\nRunning chat() - first-call compile + actual sampling...")
  t0 = time.time()
  out = sampler.chat(args.prompt, max_new_tokens=args.max_new_tokens,
                     multi_turn=False)
  print(f"  done in {time.time() - t0:.1f}s")
  print(f"  output (first 80 chars): {out[:80]!r}")

  post = hbm_snapshot()
  print_snapshot("HBM after chat() (in_use is post-call; peak is the spike)",
                 post)
  print_delta(pre, post)
  # The cache is allocated DURING chat() and may be released by the time
  # we snapshot post-call. The peak_bytes_in_use field captures the spike.
  print("\n--- Peak HBM during chat() (relative to pre) ---")
  print(f"  {'chip':>4}  {'pre_peak':>14}  {'post_peak':>14}  "
        f"{'peak_delta':>14}")
  for d_id in sorted(pre):
    pre_p = pre[d_id]["peak_bytes_in_use"]
    post_p = post[d_id]["peak_bytes_in_use"]
    print(f"  {d_id:>4}  {_abs(pre_p):>14}  {_abs(post_p):>14}  "
          f"{_human(post_p - pre_p):>14}")

  # Summary: how lopsided is the per-chip PEAK distribution?
  peak_deltas = sorted(
      (post[d]["peak_bytes_in_use"] - pre[d]["peak_bytes_in_use"]
       for d in pre),
      reverse=True,
  )
  if len(peak_deltas) >= 2:
    biggest = peak_deltas[0]
    rest_avg = sum(peak_deltas[1:]) / max(1, len(peak_deltas) - 1)
    print("\nVerdict (based on peak HBM spike during chat):")
    if biggest > 4 * max(1, abs(rest_avg)):
      print("  Cache is concentrated on one chip "
            f"(biggest peak delta = {_human(biggest)}, others avg = "
            f"{_human(rest_avg)}). Confirms Hypothesis B: cache is NOT "
            "sharded across the slice.")
    elif biggest <= 1024 * 1024:
      print("  No measurable peak delta (<1 MiB). The probe may not have "
            "captured the spike; try a larger cache_length.")
    elif all(d > 0.5 * biggest for d in peak_deltas):
      print("  Cache spike is roughly even across chips; sharding is "
            "happening somewhere (or the cache is replicated, which uses "
            "the same memory on every chip).")
    else:
      print("  Cache spike is uneven across chips. Inspect the per-chip "
            "table above to see which chip(s) absorbed it.")

  return 0


if __name__ == "__main__":
  sys.exit(main())
