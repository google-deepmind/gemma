"""End-to-end equivalence + memory test for KVCacheMode.LOCAL_WINDOW.

Each KV cache mode is benchmarked in its own subprocess so that
`peak_bytes_in_use` is a clean per-process high-water mark, not polluted by
a previous mode's allocations. The driver spawns two subprocesses, captures
their output (greedy text + per-chip peak HBM as JSON on stdout), and
reports the comparison.

Usage on TPU node:

    # Default: E2B smoke test, cache_length=4096, prompt <= window.
    python cache_local_window_test.py

    # Stress test: prompt LONGER than the window (forces ring wrap).
    python cache_local_window_test.py --variant e4b --cache_length 16384 \\
        --prompt_repeat 200

    # Stress test: long context, big cache, the win that matters.
    python cache_local_window_test.py --variant e4b --cache_length 32768

If you want to inspect a single mode interactively, use --mode:

    python cache_local_window_test.py --variant e4b --cache_length 16384 \\
        --mode local_window
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _human(b: float) -> str:
  for u in ('B', 'KiB', 'MiB', 'GiB'):
    if abs(b) < 1024:
      return f'{b:.2f} {u}'
    b /= 1024
  return f'{b:.2f} TiB'


def hbm_peak() -> dict[int, int]:
  """Per-device peak HBM. Imports JAX lazily so the driver process doesn't
  initialize JAX (and therefore doesn't grab any TPU memory)."""
  import jax  # pylint: disable=g-import-not-at-top
  out = {}
  for d in jax.devices():
    if not hasattr(d, 'memory_stats'):
      continue
    try:
      stats = d.memory_stats() or {}
    except Exception:  # pylint: disable=broad-except
      stats = {}
    out[int(d.id)] = int(stats.get('peak_bytes_in_use', 0))
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


def run_one_mode(args, mode_name: str) -> dict:
  """Run a single mode in this process. Returns a dict of results."""
  from gemma.gm.utils import _cache_helper  # pylint: disable=g-import-not-at-top
  import jax  # pylint: disable=g-import-not-at-top

  prompt = (args.prompt + ' ') * max(1, args.prompt_repeat)
  mode = _cache_helper.KVCacheMode(mode_name)

  sampler, _ = build_sampler(
      args.variant, args.cache_length, mode, args.params_fsdp,
  )
  t0 = time.time()
  out = sampler.chat(prompt, max_new_tokens=args.max_new_tokens,
                     rng=args.seed)
  elapsed = time.time() - t0
  peak = hbm_peak()
  return {
      'mode': mode_name,
      'variant': args.variant,
      'cache_length': args.cache_length,
      'output': out,
      'time_s': elapsed,
      'peak_per_chip': peak,
      'jax_devices': [str(d) for d in jax.devices()],
  }


def _print_single_mode_report(result: dict) -> None:
  print(f'\n=== mode={result["mode"]} ===')
  print(f'  variant: {result["variant"]}, cache_length: {result["cache_length"]}')
  print(f'  time:    {result["time_s"]:.1f}s')
  print(f'  output:  {result["output"][:120]!r}')
  for d, p in sorted(result['peak_per_chip'].items()):
    print(f'  chip {d} peak HBM: {_human(p)}')


def _spawn_subprocess(script_path: str, args, mode_name: str) -> dict:
  """Spawn a fresh Python that runs only one mode, parse its JSON line."""
  cmd = [
      sys.executable, script_path,
      '--mode', mode_name,
      '--variant', args.variant,
      '--cache_length', str(args.cache_length),
      '--prompt', args.prompt,
      '--prompt_repeat', str(args.prompt_repeat),
      '--max_new_tokens', str(args.max_new_tokens),
      '--seed', str(args.seed),
      '--emit_json',
  ]
  if args.params_fsdp:
    cmd.append('--params_fsdp')
  else:
    cmd.append('--no-params_fsdp')

  print(f'\n[spawning subprocess for mode={mode_name}]')
  print(f'  cmd: {" ".join(cmd)}')
  proc = subprocess.run(
      cmd, capture_output=True, text=True, check=False,
      env={**os.environ},
  )
  # Stream the child's stderr so the user sees compile progress / warnings.
  if proc.stderr:
    sys.stderr.write(proc.stderr)
  if proc.returncode != 0:
    print(f'  subprocess failed (returncode={proc.returncode})')
    print(proc.stdout)
    raise SystemExit(proc.returncode)
  # The last non-empty line of stdout is our JSON payload (the rest is
  # human-readable progress that we let the child print).
  json_line = None
  for line in proc.stdout.splitlines():
    line = line.strip()
    if line.startswith('{') and line.endswith('}'):
      json_line = line
  if json_line is None:
    print(proc.stdout)
    raise RuntimeError('subprocess did not emit a JSON result line')
  return json.loads(json_line)


def main(argv=None) -> int:
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
  parser.add_argument(
      '--mode', choices=('legacy', 'local_window'), default=None,
      help='If set, run only this mode in this process and emit a one-line '
           'JSON result. Used by the driver to fork clean subprocesses; you '
           'can also set it manually for single-mode runs.',
  )
  parser.add_argument(
      '--emit_json', action='store_true',
      help='When --mode is set, also emit a JSON line with the result on '
           'stdout (for the driver to parse).',
  )
  args = parser.parse_args(argv)

  if args.mode is not None:
    # Subprocess-mode: run one cache mode, optionally emit JSON.
    result = run_one_mode(args, args.mode)
    _print_single_mode_report(result)
    if args.emit_json:
      print(json.dumps(result))
    return 0

  # Driver: spawn one subprocess per mode for clean peak HBM, then compare.
  script_path = os.path.abspath(__file__)
  legacy = _spawn_subprocess(script_path, args, 'legacy')
  local = _spawn_subprocess(script_path, args, 'local_window')

  print('\n' + '=' * 70)
  print('Equivalence check:')
  if legacy['output'] == local['output']:
    print('  PASS: greedy outputs match exactly.')
    print(f'    output: {legacy["output"][:120]!r}')
  else:
    print('  FAIL: outputs DIFFER.')
    print(f'    LEGACY       : {legacy["output"]!r}')
    print(f'    LOCAL_WINDOW : {local["output"]!r}')

  print('\nMemory comparison (per-chip peak HBM, separate-process):')
  print(f'  {"chip":>4}  {"LEGACY":>14}  {"LOCAL_WINDOW":>14}  {"saved":>14}')
  total_saved = 0
  for d in sorted(int(k) for k in legacy['peak_per_chip']):
    leg = int(legacy['peak_per_chip'][str(d)])
    lw = int(local['peak_per_chip'].get(str(d), 0))
    saved = leg - lw
    total_saved += saved
    sign = '+' if saved >= 0 else ''
    print(f'  {d:>4}  {_human(leg):>14}  {_human(lw):>14}  '
          f'{sign}{_human(saved):>13}')
  print(f'\n  total peak HBM saved across slice: '
        f'{"+" if total_saved >= 0 else ""}{_human(total_saved)}')

  if total_saved <= 1024 * 1024:  # < 1 MiB
    print('\n  Note: total saved is in the noise. Probable causes:')
    print('    * E2B has only 1 KV head and a 512 window, so the predicted '
          'local-cache savings at L=4096 is only ~100 MiB total - small '
          'next to the ~5-6 GiB peak from FSDP-loaded params + activations.')
    print('    * Try --variant e4b --cache_length 16384 (or 32768) to get a '
          'cache big enough for the savings to be measurable.')

  print(f'\n  time:    LEGACY={legacy["time_s"]:.1f}s  '
        f'LOCAL_WINDOW={local["time_s"]:.1f}s')
  return 0 if legacy['output'] == local['output'] else 1


if __name__ == '__main__':
  sys.exit(main())
