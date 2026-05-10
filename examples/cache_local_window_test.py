"""End-to-end equivalence + memory test for KVCacheMode.LOCAL_WINDOW.

Runs the same prompt+seed through `ChatSampler.chat()` twice (LEGACY then
LOCAL_WINDOW) and:
  1. Verifies the greedy token sequences match exactly.
  2. Reports per-chip peak HBM for both modes so the win is visible.

Usage on TPU node:

    # Default: E2B smoke test, cache_length=4096, prompt <= window.
    python cache_local_window_test.py

    # Stress test: prompt LONGER than the window (forces ring wrap).
    python cache_local_window_test.py --variant e4b --cache_length 16384 \
        --prompt_repeat 200

    # Stress test: cache_length larger than window, exercise the cap.
    python cache_local_window_test.py --variant e4b --cache_length 32768

Flags:
    --variant      e2b | e4b | 31b
    --cache_length int (default 4096)
    --prompt       str (default a short greeting)
    --prompt_repeat N - repeat the prompt N times to make it longer than
                       sliding_window_size.
    --max_new_tokens int (default 32)
    --params_fsdp / --no-params_fsdp (default on)
    --tolerance_bf16 float - max allowed |logit delta|. Default 5e-2 because
                       BF16 reduction order changes between modes.
"""

from __future__ import annotations

import argparse
import sys
import time

import jax
import jax.numpy as jnp


def _human(b: float) -> str:
  for u in ('B', 'KiB', 'MiB', 'GiB'):
    if abs(b) < 1024:
      return f'{b:.2f} {u}'
    b /= 1024
  return f'{b:.2f} TiB'


def hbm_peak() -> dict[int, int]:
  out = {}
  for d in jax.devices():
    if not hasattr(d, 'memory_stats'):
      continue
    try:
      stats = d.memory_stats() or {}
    except Exception:  # pylint: disable=broad-except
      stats = {}
    out[d.id] = int(stats.get('peak_bytes_in_use', 0))
  return out


def build_sampler(variant: str, cache_length: int, kv_cache_mode,
                  params_fsdp: bool, params=None):
  from gemma import gm  # pylint: disable=g-import-not-at-top

  variants = {
      'e2b': (gm.nn.Gemma4_E2B, 'GEMMA4_E2B_IT'),
      'e4b': (gm.nn.Gemma4_E4B, 'GEMMA4_E4B_IT'),
      '31b': (gm.nn.Gemma4_31B, 'GEMMA4_31B_IT'),
  }
  cls, ckpt_name = variants[variant.lower()]
  model = cls()
  if params is None:
    ckpt_path = getattr(gm.ckpts.CheckpointPath, ckpt_name)
    if params_fsdp:
      from kauldron import kd  # pylint: disable=g-import-not-at-top
      params = gm.ckpts.load_params(
          ckpt_path, sharding=kd.sharding.FSDPSharding()
      )
    else:
      params = gm.ckpts.load_params(ckpt_path)
  sampler = gm.text.ChatSampler(
      model=model,
      params=params,
      cache_length=cache_length,
      kv_cache_mode=kv_cache_mode,
      multi_turn=False,
  )
  return sampler, params


def main(argv=None) -> int:
  from gemma.gm.utils import (  # pylint: disable=g-import-not-at-top
      _cache_helper,
  )

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--variant', default='e2b', choices=('e2b', 'e4b', '31b'))
  parser.add_argument('--cache_length', type=int, default=4096)
  parser.add_argument('--prompt', default='Tell me a one-sentence fun fact.')
  parser.add_argument('--prompt_repeat', type=int, default=1,
                      help='Repeat the prompt this many times to stress the '
                           'window-wrap path (use >= 200 for E4B with '
                           'sliding_window=512 to force wrap).')
  parser.add_argument('--max_new_tokens', type=int, default=32)
  parser.add_argument('--params_fsdp',
                      action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args(argv)

  prompt = (args.prompt + ' ') * max(1, args.prompt_repeat)

  print(f'JAX devices: {jax.devices()}')
  print(f'Model: {args.variant.upper()}, cache_length={args.cache_length}')
  print(f'Prompt length (chars): {len(prompt)}')
  print(f'max_new_tokens: {args.max_new_tokens}')
  print('-' * 70)

  # ---- Run 1: LEGACY ---- (load params here, share with run 2)
  print('\n[1/2] Running LEGACY (current behavior)...')
  sampler_legacy, params = build_sampler(
      args.variant, args.cache_length,
      _cache_helper.KVCacheMode.LEGACY, args.params_fsdp,
  )
  t0 = time.time()
  out_legacy = sampler_legacy.chat(
      prompt, max_new_tokens=args.max_new_tokens, rng=args.seed,
  )
  legacy_time = time.time() - t0
  legacy_peak = hbm_peak()
  print(f'  output: {out_legacy[:120]!r}')
  print(f'  time:   {legacy_time:.1f}s')
  for d, p in sorted(legacy_peak.items()):
    print(f'  chip {d} peak HBM: {_human(p)}')

  del sampler_legacy

  # ---- Run 2: LOCAL_WINDOW ----
  print('\n[2/2] Running LOCAL_WINDOW...')
  sampler_lw, _ = build_sampler(
      args.variant, args.cache_length,
      _cache_helper.KVCacheMode.LOCAL_WINDOW, args.params_fsdp, params=params,
  )
  t0 = time.time()
  out_lw = sampler_lw.chat(
      prompt, max_new_tokens=args.max_new_tokens, rng=args.seed,
  )
  lw_time = time.time() - t0
  lw_peak = hbm_peak()
  print(f'  output: {out_lw[:120]!r}')
  print(f'  time:   {lw_time:.1f}s')
  for d, p in sorted(lw_peak.items()):
    print(f'  chip {d} peak HBM: {_human(p)}')

  # ---- Compare ----
  print('\n' + '=' * 70)
  print('Equivalence check:')
  if out_legacy == out_lw:
    print(f'  PASS: greedy outputs match exactly.')
  else:
    print(f'  FAIL: outputs DIFFER.')
    print(f'    LEGACY:       {out_legacy!r}')
    print(f'    LOCAL_WINDOW: {out_lw!r}')

  print('\nMemory comparison (peak HBM during chat):')
  print(f'  {"chip":>4}  {"LEGACY":>14}  {"LOCAL_WINDOW":>14}  {"saved":>14}')
  total_saved = 0
  for d in sorted(legacy_peak):
    saved = legacy_peak[d] - lw_peak.get(d, 0)
    total_saved += saved
    print(f'  {d:>4}  {_human(legacy_peak[d]):>14}  '
          f'{_human(lw_peak.get(d, 0)):>14}  {_human(saved):>14}')
  print(f'  total peak HBM saved across slice: {_human(total_saved)}')

  return 0 if out_legacy == out_lw else 1


if __name__ == '__main__':
  sys.exit(main())
