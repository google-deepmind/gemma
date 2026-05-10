# Gemma JAX KV Cache: Memory & Sharding Design Proposal

**Author:** Karl (with Claude Code assistance)
**Status:** Updated after design review
**Target:** `github.com/google-deepmind/gemma` (this repo)
**Date:** 2026-05-09
**Related code paths:**
- [`gemma/gm/text/_sampler.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_sampler.py)
- [`gemma/gm/text/_chat_sampler.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_chat_sampler.py)
- [`gemma/gm/text/_gemma4_sampler.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_gemma4_sampler.py)
- [`gemma/gm/text/_prefill.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_prefill.py)
- [`gemma/gm/text/_sampler_loop.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_sampler_loop.py)
- [`gemma/gm/nn/gemma4/_config.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_config.py)
- [`gemma/gm/nn/gemma4/_modules.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_modules.py)
- [`gemma/gm/nn/gemma4/_transformer.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_transformer.py)
- [`gemma/gm/utils/_cache_helper.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/utils/_cache_helper.py)

---

## 1. Problem Statement

The `gm.text.ChatSampler` / `gm.text.Gemma4Sampler` inference path is
substantially less HBM-efficient than necessary on multi-chip TPU slices,
specifically for the Gemma 4 model family which uses hybrid
local-sliding + global attention. Two independent issues compound:

**Problem A — Eager, uniform cache allocation across all layers.**
At sampler construction time (more precisely at the start of every
non-multi-turn prefill call), the cache is allocated as a fixed
`[batch, cache_length, num_kv_heads, head_dim]` buffer **identically
sized on every layer**, including layers whose attention type is
`LOCAL_SLIDING` and which only ever read the most recent
`sliding_window_size` tokens during causal decode
([`gemma4/_modules.py:343-349`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_modules.py#L343-L349)
masks all positions outside the window). For a Gemma 4 model that uses a
5L:1G pattern (E4B has 35 local + 7 global; 31B has 50 local + 10 global;
26B-A4B has 25 local + 5 global), most of the per-layer cache budget is
zeroed bytes that the attention mask discards. The waste fraction grows
linearly with `cache_length` until at long contexts the cache is
dominated by guaranteed-unread bytes.

**Problem B — Cache is replicated across the device slice.**
At all current call sites
([`gemma4/_transformer.py:412`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_transformer.py#L412),
[`_transformer.py:329`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/_transformer.py#L329),
[`gemma3n/_transformer.py:370`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma3n/_transformer.py#L370))
the cache pytree is passed to
`kd.sharding.with_sharding_constraint(cache, sharding)` with no
intrinsic, model-aware partition spec. The user-facing samplers
(`Sampler`, `Gemma4Sampler`, `ChatSampler`) accept a `sharding=` kwarg
but the documented usage path
([`colabs/sharding.ipynb`](https://github.com/google-deepmind/gemma/blob/main/colabs/sharding.ipynb))
does not pass one. With `sharding=None` the constraint is a no-op and
the cache lives wherever the active mesh / JIT default puts it. With
FSDP-sharded params and no explicit cache sharding, the cache gets
**replicated across every chip** — same memory cost on every chip in
the slice, scaling 1× rather than 1/N×.

The combination is a multiplicative loss: every chip pays for cache
slots that are never read. On memory-constrained slices (16 GB / chip
on TPU v5e) it is the dominant blocker for running the larger Gemma 4
variants (31B, 26B-A4B) at any usable context length, and a soft
blocker for E2B/E4B at long contexts (≥16K).

### 1.1. Empirical evidence

Two scripts ([`examples/cache_audit.py`](./examples/cache_audit.py) and
[`examples/cache_probe.py`](./examples/cache_probe.py), included in
this proposal) collect the following data on a v5e-4 slice (4 chips,
16 GB HBM each, FSDP-sharded params):

**Cache audit, Gemma4_E4B (35 LOCAL_SLIDING + 7 GLOBAL layers,
`num_kv_heads=2`, `head_dim=256` local / 512 global,
`sliding_window_size=512`):**

| `cache_length` | Total cache bytes | LOCAL bytes | GLOBAL bytes | Ratio vs L=256 |
|---:|---:|---:|---:|---:|
|   256 |   24.54 MiB |  17.53 MiB |   7.01 MiB | 1.00× |
| 4 096 |  392.66 MiB | 280.55 MiB | 112.11 MiB | 16.00× |
|16 384 | 1 506.11 MiB | 1 100 MiB | 448 MiB | 64.00× |

The ratio is **exactly linear in `cache_length` across every layer**,
including all 35 local-sliding layers. With a `sliding_window_size=512`,
the local layers only ever read `<= 512` slots; at `cache_length=16384`
the local-attention storage is **31× over-allocated per local layer**
(15 872 unread slots for every 512 read).

**Runtime probe, Gemma4_E4B, ChatSampler with FSDP-sharded params,
single chat() turn (~25 output tokens):**

| Metric | L=4 096 | L=16 384 |
|---|---:|---:|
| HBM after params load (per chip) | 7.81 GiB | 7.81 GiB |
| Peak HBM during chat (chip 0) | 9.46 GiB | 14.04 GiB |
| Peak HBM during chat (chips 1-3) | 9.37 GiB | 13.95 GiB |
| **Per-chip peak Δ** | **+1.6 GiB** | **+6.2 GiB** |
| Per-chip post-chat Δ (steady state) | +800 MiB | +3.08 GiB |

Interpretation:

1. The peak deltas are roughly equal across all 4 chips (chip 0 is only
   ~90 MiB above the others). This **confirms Problem B with replication
   semantics**: every chip absorbs the full cache cost.
2. The post-chat steady-state delta of ~800 MiB at L=4096 is roughly
   **2× the logical cache size** (392 MiB). The extra ~400 MiB is the
   `last_state.cache` held by `ChatSampler` plus a transient second
   copy created by the functional update in `_merge_cache`
   ([`_prefill.py:357-366`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_prefill.py#L357-L366)).
3. At L=16384 the per-chip peak exceeds 14 GiB out of a 15.75 GiB chip
   limit. With ~6 GiB of headroom for activations/intermediate buffers
   already consumed, this is the regime where small jit-time
   re-allocations OOM — matching the originally reported
   "32 MB allocation fails with chip 0 at near-zero free" symptom on
   the bigger models (31B / 26B-A4B).

### 1.2. Why this matters for production-style usage on TPU v5e

The repo's stated audience for this sampler is research/experimentation
([`gm.text.Sampler` docstring](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_sampler.py#L86)
explicitly recommends `gm.text.ChatSampler` "for most use cases"). But
the sampler is the only ergonomic in-repo path to evaluate Gemma 4 on a
TPU slice without standing up a separate inference engine. Closing
even half the gap between the current implementation and a
production-quality KV cache layout (à la MaxText / JetStream) makes
the official sampler a viable evaluation tool for the larger Gemma 4
variants on commodity TPU slices.

### 1.3. Out of scope

The following are intentionally NOT in scope for this proposal:

- **PagedAttention / vLLM-style block-table caches.** Significant
  surgery (block tables, Pallas gather kernels, ragged attention math).
  Only a clear win for batched, multi-tenant serving with
  heterogeneous request lengths. For a research sampler with batch=1
  the eager-buffer approach is fine *once* Problems A and B are fixed.
- **Quantized KV cache** (int8 / fp8). Independent axis of
  improvement; can be layered on top of either fix.
- **Cross-sampler refactor** (unifying `Sampler` / `Gemma4Sampler` /
  `ChatSampler`). This proposal touches their shared dependencies but
  preserves their public interfaces.

---

## 2. Code Archaeology

### 2.1. Cache data structure

The cache is a **plain `dict[str, dict[str, jax.Array]]`** keyed by
`f"layer_{i}"`, with no axis-aware annotations:

```
type alias:  Cache = dict[str, _modules.LayerCache]
type alias:  LayerCache = dict[str, jax.Array]
```

Defined at
[`gemma/gm/nn/_config.py:28`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/_config.py#L28)
(legacy) and
[`gemma/gm/nn/gemma4/_config.py:29`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_config.py#L29)
(Gemma 4). Per layer, four arrays:

| Field | Shape | Dtype | Role |
|---|---|---|---|
| `'k'` | `[B, cache_size, num_kv_heads, head_dim]` | bf16 | key cache |
| `'v'` | `[B, cache_size, num_kv_heads, head_dim]` | bf16 | value cache |
| `'positions'` | `[B, cache_size]` | int32 | per-slot RoPE position (used by sliding mask) |
| `'end_index'` | `[B]` | int32 | write pointer (mod cache_size) |

A thin slicing wrapper exists
([`_cache_helper.Cache`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/utils/_cache_helper.py#L31-L89))
but it just wraps the dict — it doesn't change the underlying layout.

### 2.2. Allocation and sharding

**Single-shot allocation per first-turn `sample()` / `chat()` call.**
The full call chain for Gemma 4:

1. User → `Gemma4Sampler.sample()` /
   `ChatSampler.chat()` ([`_chat_sampler.py:261`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_chat_sampler.py#L261)).
2. → `_prefill.prefill(... cache_length=self.cache_length, sharding=sharding)` ([`_gemma4_sampler.py:206`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_gemma4_sampler.py#L206)).
3. → `_get_or_init_cache()` ([`_prefill.py:283-307`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_prefill.py#L283-L307)).
4. → `model.init_cache(...)` ([`_prefill.py:295-300`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_prefill.py#L295-L300)).
5. → `Gemma4Transformer.init_cache` ([`gemma4/_transformer.py:399-412`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_transformer.py#L399-L412), `nn.jit`-decorated, static_argnames includes `cache_length`, `sharding`, etc.).
6. → `TransformerConfig.init_cache` ([`gemma4/_config.py:157-193`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_config.py#L157-L193)).
7. → `_modules.Attention.init_cache` ([`gemma4/_modules.py:397-417`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_modules.py#L397-L417)) — `jnp.zeros(...)` per layer.

The sharding plumbing is a single
`kd.sharding.with_sharding_constraint(cache, sharding)` over the entire
pytree at
[`gemma4/_transformer.py:412`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_transformer.py#L412):

```python
return kd.sharding.with_sharding_constraint(cache, sharding)
```

If the user does not pass a `sharding=` argument (the documented
pattern), this is a no-op. The cache then takes the JAX default
placement, which inside an active FSDP mesh is **replicated** (as
confirmed by the runtime probe).

**`cache_length` source.** A user-tunable field on the sampler, default
4096
([`_sampler.py:137`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_sampler.py#L137),
[`_gemma4_sampler.py:76`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_gemma4_sampler.py#L76),
[`_chat_sampler.py:127`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_chat_sampler.py#L127)).
Used as a single integer argument applied uniformly to every layer in
[`gemma4/_config.py:171-192`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_config.py#L171-L192).

**Allocation frequency.** Once per first-turn `sample()` / `chat()`. In
multi-turn mode, the cache is reused across turns
([`_prefill.py:301-303`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_prefill.py#L301-L303):
`cache = prev_turns.cache`).

### 2.3. The local-sliding waste, exposed

The structural location of Problem A is
[`gemma4/_config.py:171-192`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_config.py#L171-L192):

```python
for i, attn_type in enumerate(self.attention_types):
  if (attn_type == _modules.AttentionType.GLOBAL
      and self.global_key_size is not None):
    cache[f'layer_{i}'] = _modules.Attention.init_cache(
        cache_length,                                            # uniform
        self.num_global_kv_heads if self.num_global_kv_heads
        else self.num_kv_heads,
        self.global_key_size,
        batch_size, dtype)
  else:
    cache[f'layer_{i}'] = _modules.Attention.init_cache(
        cache_length,                                            # uniform
        self.num_kv_heads,
        self.head_dim,
        batch_size, dtype)
```

The two branches differ in `num_kv_heads` and `head_dim` only. The
`cache_length` argument is identical for global and local-sliding
layers. The mask in
[`gemma4/_modules.py:338-349`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_modules.py#L338-L349)
already discards everything outside the window — no slot beyond
`sliding_window_size` is ever attended to in a local-sliding layer.

The same pattern exists in
[`gemma3n/_config.py:178-194`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma3n/_config.py#L178-L194).
The original Gemma 1-3 path
([`gemma/gm/nn/_config.py:120-142`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/_config.py#L120-L142))
is uniform but Gemma 1-3 don't have hybrid attention, so it's not a
loss there.

### 2.4. Decode-step writes

Each decode step writes one slot via `dynamic_update_slice` /
`array.at[...].set(...)` into the pre-allocated buffer; there is no
physical growth. The write index uses modular wrap (`end_index %
cache_size` at
[`gemma4/_modules.py:304`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_modules.py#L304))
but the loop halts before the wrap matters — `cond_fn` checks
`is_full = end_index >= total_cache_length - 1`
([`_cache_helper.py:86-89`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/utils/_cache_helper.py#L86-L89),
[`_sampler_loop.py:160-171`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_sampler_loop.py#L160-L171)).
The mod-wrap is defensive, not load-bearing.

### 2.5. The "is_full" first-layer assumption

`Cache.total_cache_length` reads `next(iter(self.cache.values()))['k'].shape[1]`
([`_cache_helper.py:43-47`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/utils/_cache_helper.py#L43-L47)).
It assumes every layer has the same cache length. **Any change that
introduces per-layer cache sizes must update this method** to either
(a) return the max across layers, or (b) be replaced with per-layer
queries.

---

## 3. Proposed Changes

This proposal includes **two independent but composable changes**,
ranked by HBM impact at long context. Either can be merged
independently.

### 3.1. Change A: right-size LOCAL_SLIDING cache to the window

**Summary.** Cap each local-sliding layer's cache size at
`min(cache_length, sliding_window_size)`. Global layers retain the
full `cache_length`.

The implementation should be toggled with `KVCacheMode`, defaulting to
`LEGACY`, and prefill should expose `KVPrefillMode.LEGACY_SCRATCH`
through `GEMMA_KV_PREFILL_MODE=legacy_scratch` so benchmarks can record
the active prefill strategy explicitly.

**Why not "+1".** The window mask at
[`gemma4/_modules.py:50-51`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_modules.py#L50-L51)
uses strict bounds: `cache_position > position - sliding_window_size`
and `cache_position < position + sliding_window_size`. With the normal
causal mask during decode, that admits the current token plus the
previous `sliding_window_size - 1` tokens. The persistent causal local
cache therefore needs exactly `sliding_window_size` slots, not
`sliding_window_size + 1`.

**HBM saved (E4B, batch=1, bf16, per chip if currently replicated):**

| `cache_length` | Current local cache | After change | Savings per chip |
|---:|---:|---:|---:|
|   4 096 | 280 MiB |  35 MiB | **245 MiB** |
|  16 384 | 1.10 GiB |  35 MiB | **1.07 GiB** |
|  32 768 | 2.19 GiB |  35 MiB | **2.16 GiB** |
| 131 072 | 8.75 GiB |  35 MiB | **8.72 GiB** |

(Local-cache size becomes constant once `cache_length >= window`. The
~35 MiB figure for E4B is `35 layers x 1 x 512 x 2 x 256 x 2(k+v) x 2 B`.)

For 31B (50 local + 10 global, sliding_window=1024, num_kv_heads=16
local / 4 global): savings scale by ~ `50/35 × 16/2 ≈ 11×` versus E4B
in absolute terms.

**Implementation.**

Patch [`gemma/gm/nn/gemma4/_config.py:157-193`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_config.py#L157-L193):

```python
def init_cache(self, batch_size, dtype=jnp.bfloat16, *, cache_length):
  if cache_length is None:
    raise ValueError('Missing `cache_length=` kwarg when calling `init_cache()`.')

  # NEW: Local-sliding layers only ever read sliding_window_size slots.
  # Cap their cache to that to avoid allocating bytes that are guaranteed
  # to be masked out at attention time.
  local_cache_length = (
      min(cache_length, self.sliding_window_size)
      if self.sliding_window_size is not None
      else cache_length
  )

  cache: Cache = {}
  for i, attn_type in enumerate(self.attention_types):
    if (attn_type == _modules.AttentionType.GLOBAL
        and self.global_key_size is not None):
      cache[f'layer_{i}'] = _modules.Attention.init_cache(
          cache_length,                                  # full
          self.num_global_kv_heads or self.num_kv_heads,
          self.global_key_size,
          batch_size, dtype)
    else:
      cache[f'layer_{i}'] = _modules.Attention.init_cache(
          local_cache_length,                            # NEW: window-capped
          self.num_kv_heads,
          self.head_dim,
          batch_size, dtype)
  return cache
```

Do Gemma 4 first. Mirror into Gemma 3n only after the Gemma 4 path has
passed equivalence tests, because Gemma 4 is the critical long-context
path and already exercises heterogeneous local/global head dimensions.

**Required collateral changes.** Per-layer cache sizes now vary, so
several pieces of code that assume a uniform layer cache need to be
fixed:

1. **`Cache.total_cache_length`**
   ([`_cache_helper.py:43-47`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/utils/_cache_helper.py#L43-L47)).
   Currently reads layer 0's `k.shape[1]`. Must return either the max
   across layers or be replaced with a per-layer query. Recommendation:
   `return max(d['k'].shape[1] for d in self.cache.values())`. The
   `is_full` semantic at line 86-89 should then use the max — i.e.,
   "stop sampling when the *largest* (typically global) cache is full",
   not when the local cache fills up at the sliding-window boundary.

2. **Full logical masks plus physical local-slot mapping.**
   `SamplingState.attention_mask_for_step` should continue to build a
   mask at the full logical cache length. Local attention cannot rely
   on implicit einsum truncation once the local cache is ring-buffered:
   physical slot `s` no longer means logical token `s`. Local cache
   entries need `logical_index` and `valid` metadata, and
   `Attention.__call__` must gather the logical mask onto physical
   slots before applying the sliding mask.

3. **Prefill prompt longer than `sliding_window_size`.** Do not simply
   pass a window-sized local cache into prefill. The attention module
   writes K/V into the supplied cache before computing attention; a
   window-sized prefill cache would overwrite early prompt K/V before
   later prompt tokens and later global layers can use them. The safe
   first implementation is:
   - Allocate the persistent decode cache with local layers sized to
     `sliding_window_size`.
   - Build a full-size prefill scratch cache for local layers, with no
     local-window metadata, so prefill attention has legacy semantics.
   - Run `model.apply` against the scratch.
   - Compact the returned local layers back into the persistent ring
     cache.
   - Copy returned global layers into the persistent global cache.

4. **Compaction must be per batch row.** A naive "last W logical slots"
   compaction is wrong for padded batches: a short row can have live
   local-context prompt tokens near logical slots 0..N while another
   row forced the bucket to a much larger logical length. The safe rule
   is to keep the last W valid logical slots per batch row, place each
   by `absolute_position % W`, and store the original logical slot in
   `logical_index` for future mask gathers.

5. **Decode writes must evict by position, not padded logical index.**
   Local-window decode writes should use `segment_pos % window` for the
   physical slot and store `end_index + arange(seq_len)` only as
   `logical_index` metadata. This preserves the local sliding window
   for padded rows while keeping the sampler's full logical mask
   intact.

**Risks.**

- **Correctness of scratch compaction.** Need to verify that ROPE
  positions, `cache_positions`, `logical_index`, and `valid` are
  correct after the prefill+merge dance. The sliding mask at
  [`gemma4/_modules.py:343-349`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_modules.py#L343-L349)
  uses `cache_positions` (the actual position stored at each slot), so
  as long as we write the correct `positions` alongside the K/V values
  and gather the logical mask through `logical_index`, the ring-buffer
  math handles wraparound. Add regression tests for prompt lengths
  `< window`, `== window`, and `> window`, plus padded batches where
  one row is much shorter than the bucket length.
- **Recompilation.** Changing layer shapes triggers a JIT recompile.
  Acceptable, one-time.
- **`is_full` semantics.** Today the loop stops when *layer 0* fills.
  Layer 0 is local-sliding in every Gemma 4 variant, so today the
  loop currently stops when the local cache fills (around step 512), which
  is *the wrong stopping condition* for global layers. Today this
  isn't observed because users typically pick `cache_length ≥ 4096 ≫
  sliding_window` and don't notice. After this change, with the local
  cache *physically* sized to the window, the same `is_full` would fire
  at step 512 always. The sampler loop must stop on the logical
  `SamplerLoop.cache_length` or `state.full_attention_mask.shape[-1]`,
  not on any local layer's physical shape.

**Test plan.**

1. **Unit test:** for every Gemma 4 variant, verify
   `model.init_cache(cache_length=4096)` produces local-layer cache
   shapes of `[1, sliding_window_size, num_kv_heads, head_dim]` and
   global-layer caches of `[1, 4096, num_global_kv_heads or
   num_kv_heads, global_key_size or head_dim]`.
2. **Numerical equivalence:** run a fixed prompt with a fixed RNG seed
   through `ChatSampler.chat()` before and after the change. Output
   tokens should match for greedy sampling. Logits only need BF16
   tolerance because compacting/gathering can change reduction order.
3. **HBM regression:** run [`examples/cache_probe.py`](./examples/cache_probe.py)
   at L=4096 and L=16384 and verify the per-chip steady-state delta
   drops as predicted in §3.1's table.

### 3.2. Change B: cache sharding contract

**Summary.** Add a model-level contract that exposes a
`PartitionSpec` pytree for the cache, computed from the model config
and the active mesh. Apply it inside `init_cache` regardless of
whether the user passed a `sharding=` kwarg. For batch=1 decode, shard
along `num_kv_heads` when the head count is divisible by the mesh's TP
axis size; replicate otherwise.

**Why this and not "expose more knobs to the user".** Today's
`sharding=` kwarg is a generic `kd.sharding.ShardingTree`. The user has
no documented way to construct one for the cache (no example exists in
the repo, and `kd.sharding.FSDPSharding()` — the only documented
sharding helper — shards axis 0, which is `batch` in the cache, which
is 1 at decode time). The cache *is* a model-internal data structure
with a model-internal layout; the model is the right place to know how
it should partition.

**HBM saved (E4B at L=4096, currently 392 MiB replicated on every chip
of a 4-chip slice = 1.57 GiB total slice HBM):**

- E4B (`num_kv_heads=2`, 4 chips → TP=2 + replicate=2): each chip
  holds 1/2 of the cache → 196 MiB / chip → **196 MiB saved per chip**
  (vs. 392 MiB replicated). Total slice HBM drops from 1.57 GiB → 784
  MiB.
- 31B (`num_kv_heads=16` local, 4 global, 4 chips → TP=4): each chip
  holds 1/4 → **3/4 saved per chip** for local layers; global layers
  shard exactly 4-ways. Roughly 4× per-chip cache reduction.
- 26B-A4B (`num_kv_heads=8` local, 2 global, 4 chips → TP=4 for local,
  TP=2+rep for global): roughly 4× for local, 2× for global.
- E2B (`num_kv_heads=1`, can't shard heads): no win — falls back to
  replication (current behavior). Acceptable.

Composes multiplicatively with Change A. Combined for 31B at
`cache_length=32768`:

- Today: ~30 GiB replicated, OOMs immediately.
- Change A only: ~4 GiB replicated.
- Change B only: ~7-8 GiB per chip.
- Change A + B: ~1 GiB per chip — fits with room for activations.

**Implementation.**

Add a method to `TransformerConfig` (Gemma 4) — and analogous changes
to Gemma 1-3 and Gemma 3n configs — that returns a cache partition
spec given the active mesh:

```python
def cache_partition_spec(self, mesh, *, tp_axis: str = 'tensor'):
  """Return a PartitionSpec pytree for the cache.

  Shards k/v along num_kv_heads when the head count is divisible by the
  TP axis size; otherwise leaves replicated. Other cache fields
  (positions, end_index) are always replicated.
  """
  from jax.sharding import PartitionSpec as P, NamedSharding
  if tp_axis not in mesh.axis_names:
    # No TP axis — replicate everything.
    pspec_kv = P()
    pspec_pos = P()
    pspec_end = P()
  else:
    tp_size = mesh.shape[tp_axis]
    # Replicate fallback if any layer's heads don't divide evenly.
    def kv_spec(num_heads):
      return P(None, None, tp_axis, None) if num_heads % tp_size == 0 else P()
    # We build a per-layer spec because num_kv_heads can vary.
    ...

  spec_tree = {}
  for i, attn_type in enumerate(self.attention_types):
    if attn_type == _modules.AttentionType.GLOBAL and self.global_key_size:
      heads = self.num_global_kv_heads or self.num_kv_heads
    else:
      heads = self.num_kv_heads
    spec_tree[f'layer_{i}'] = {
        'k': NamedSharding(mesh, kv_spec(heads)),
        'v': NamedSharding(mesh, kv_spec(heads)),
        'positions': NamedSharding(mesh, P()),
        'end_index': NamedSharding(mesh, P()),
    }
  return spec_tree
```

Then in
[`gemma/gm/nn/gemma4/_transformer.py:399-412`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_transformer.py#L399-L412):

```python
def init_cache(self, *, batch_size, dtype, cache_length, sharding=None):
  cache = self.config.init_cache(
      batch_size=batch_size, dtype=dtype, cache_length=cache_length)

  # NEW: derive a model-aware default sharding from the active mesh,
  # use it unless the user passed an explicit sharding.
  if sharding is None:
    mesh = _get_active_mesh()  # via jax.sharding.get_abstract_mesh() or similar
    if mesh is not None:
      sharding = self.config.cache_partition_spec(mesh)

  return kd.sharding.with_sharding_constraint(cache, sharding)
```

Also: add `with_sharding_constraint` after every per-step cache write
in `Attention.__call__` so the per-step `dynamic_update_slice` /
`array.at[...].set(...)` doesn't all-gather the result back to a
replicated layout. Locations:
- Gemma 4: [`gemma4/_modules.py:307-316`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_modules.py#L307-L316).
- Gemma 1-3: [`_modules.py:213-238`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/_modules.py#L213-L238).
- Gemma 3n: corresponding section in `gemma3n/_modules.py`.

**Risks.**

- **Mesh discovery.** Inside `nn.jit init_cache`, getting hold of "the
  active mesh" requires either (a) requiring the user to pass it
  explicitly, (b) using `jax.sharding.get_abstract_mesh()` (modern JAX),
  or (c) inferring it from the params via `jax.tree.leaves(params)[0].sharding.mesh`.
  Recommendation: (c), because it ensures the cache mesh matches the
  param mesh — same TP axis, same chip ordering — and (c) doesn't
  require any change to the public sampler API.
- **Per-step all-reduce.** With cache sharded by heads and queries
  replicated, the per-step attention op is `jnp.einsum('BTNH,BSNH->BTNS', q, k_sharded)`
  which produces a `BTNS` result with `N` sharded — fine, no reduce
  needed yet — and then `'BTNS,BSNH->BTNH'` produces a `BTNH` with `N`
  sharded. The output projection `'BTNH,NHD->BTD'` reduces over `N`,
  triggering an all-reduce. On v5e this is bandwidth-bound by ICI;
  empirical cost should be small (≪ 1 ms / step) but must be measured.
- **Non-divisible head counts.** E4B's `num_kv_heads=2` on a 4-chip
  slice → TP=2, replicate=2 (the audit script confirms this works).
  E2B's `num_kv_heads=1` → fall back to replication. The proposed
  helper handles both via the `kv_spec(heads)` divisibility check.
- **Backward compatibility.** Today the user can pass a
  `sharding=`. We should preserve that as an override: if the user
  passed an explicit non-`None` sharding, do not auto-derive. This
  keeps the existing API contract and lets advanced users override.

**Test plan.**

1. **Audit:** [`examples/cache_audit.py --variant {e4b,31b,26b_a4b}
   --shard heads`](./examples/cache_audit.py) already exists; verify
   the auto-shard logic in `init_cache` produces the same
   `NamedSharding(..., P(None, None, 'tensor', None))` on k/v that the
   audit script's `--shard heads` mode produces.
2. **HBM regression:** [`examples/cache_probe.py`](./examples/cache_probe.py)
   should show **per-chip peak deltas drop by ~TP×** for E4B/31B.
   E4B at L=4096 should drop from 1.6 GiB peak / chip to ~800 MiB
   peak / chip. 31B at L=4096 should drop from ~3.7 GiB peak (today)
   to ~900 MiB peak.
3. **Numerical equivalence:** logit-level comparison before/after the
   change for a fixed prompt and seed. Sharding does not change math.
4. **Compile-time / step-time microbenchmark:** measure decode steps/sec
   before and after on E4B at L=4096. Acceptance: ≤ 5% slowdown from
   the per-step all-reduce on a 4-chip v5e slice.

### 3.3. Out of scope but worth noting: the transient 2× cache copy

The runtime probe shows a steady-state delta of ~800 MiB at L=4096 —
about 2× the logical 392 MiB cache. The extra copy is created in
`_merge_cache` at
[`_prefill.py:357-366`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_prefill.py#L357-L366):

```python
return full_cache.at[:, : prefill_cache.total_cache_length].set_kv(prefill_cache)
```

`Cache.at[].set_kv()` uses `jnp.ndarray.at[...].set(...)` semantics
([`_cache_helper.py:113-119`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/utils/_cache_helper.py#L113-L119),
[`_cache_helper.py:149-155`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/utils/_cache_helper.py#L149-L155)),
which is functional. Without `donate_argnums` on the surrounding jit,
JAX may keep both the old and new buffers alive briefly, causing a
~2× transient. After the merge, `last_state.cache` retains the
*new* buffer; the old `full_cache` should be garbage-collected. The
fact that the steady-state delta is 2× suggests the old buffer is
still being held — likely because it's the input to `_merge_cache`
inside the JITted prefill, and JAX keeps it alive until the prefill
function returns.

This isn't part of this proposal, but a follow-up could:
- Donate the input cache buffer in the prefill jit
  (`jax.jit(prefill_fn, donate_argnums=...)`).
- Or restructure prefill to write directly into `full_cache` without
  the intermediate small-cache prefill + merge.

Estimated savings: an additional ~1× cache (cuts the per-chip
steady-state delta from 2×cache to 1×cache).

### 3.4. Stack rank

If only one change can be merged: **Change A**. It's smaller, safer,
benefits every Gemma 4 variant including E2B, and saves the most
absolute bytes at long contexts. Change B has a hard ceiling on E2B
(`num_kv_heads=1` is not shardable) and is most valuable on the bigger
models that already need the 8x slice.

If both can be merged: ship them together. They compose
multiplicatively for HBM and touch nearby code. The combined
expected reduction in per-chip cache HBM:

| Model | L | Today (replicated) | A only | B only | A + B |
|---|---:|---:|---:|---:|---:|
| E4B  |  4 096 | 392 MiB | 147 MiB | 196 MiB | 74 MiB |
| E4B  | 16 384 | 1.51 GiB | 483 MiB | 753 MiB | 241 MiB |
| 31B  |  4 096 | 3.69 GiB | 1.17 GiB | 920 MiB | 290 MiB |
| 31B  | 32 768 | 29.5 GiB | 3.52 GiB | 7.4 GiB | 880 MiB |

(Absolute numbers per chip, batch=1, bf16, 4-chip slice for E4B and
31B. 31B at L=32K with current code does not fit; with both changes,
fits with substantial headroom.)

---

## 4. Risks & Open Questions

1. **`is_full` semantics.** Currently a soft bug today (loop stops
   when layer-0 fills, which today is local-sliding). Change A makes
   this strictly observable. Fix in scope of Change A, but worth
   reviewer eyes.
2. **Mesh discovery for Change B.** Recommend inferring from
   `params` to avoid public-API churn; reviewer may prefer an explicit
   API.
3. **Per-step all-reduce cost on Change B.** Need empirical
   measurement before claiming step-time parity. If the all-reduce
   exceeds ~5% of step time, consider replicating queries after the
   sharded attention output (or using `jax.lax.psum_scatter` /
   `with_sharding_constraint` to keep the reduce out of the critical
   path).
4. **Multi-turn correctness.** Multi-turn reuses
   `prev_turns.cache`. After Change A, the cache shape is
   `{layer_i: [B, layer_cache_size, ...]}` per layer. Code paths that
   inspect cache shape (e.g.
   [`_chat_sampler._remove_eos_token`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_chat_sampler.py#L445)
   touches `cache_info` but only `end_index`) should be audited.
5. **Compatibility with `KVCacheSharingConfig`** for the
   shared-layer path
   ([`gemma4/_config.py:32-84`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_config.py#L32-L84),
   used by E2B and E4B with `frac_shared_layers > 0`). The sharing
   pattern shares cache between layers — Change A's per-layer sizes
   must be consistent across shared layer pairs (which they are, since
   sharing groups layers of the same attention type).
6. **Quantization / LoRA.** Out of scope, but worth noting that the
   cache lives at sampling time (no params-quantization interaction)
   and isn't itself quantized.

---

## 5. Plan of Work

Phased so each phase is independently revertible:

**Phase 0 (this proposal).** Land
[`examples/cache_audit.py`](./examples/cache_audit.py) and
[`examples/cache_probe.py`](./examples/cache_probe.py). They are
read-only diagnostic tools, useful regardless of which fix lands.

**Phase 1 (Change A).** Right-size local-sliding cache.

1. Add per-layer cache sizing in
   [`gemma/gm/nn/gemma4/_config.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_config.py)
   and [`gemma/gm/nn/gemma3n/_config.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma3n/_config.py).
2. Fix `Cache.total_cache_length` and `is_full` in
   [`gemma/gm/utils/_cache_helper.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/utils/_cache_helper.py).
3. Fix prefill cache slice / merge in
   [`gemma/gm/text/_prefill.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_prefill.py)
   to handle per-layer sizes.
4. Add unit tests + numerical-equivalence test.
5. Verify HBM reduction with `examples/cache_probe.py` on E4B at
   L=4096 and L=16384.

Estimated diff: ~80 LOC of source + ~150 LOC of tests.

**Phase 2 (Change B).** Cache sharding contract.

1. Add `cache_partition_spec(mesh)` to `TransformerConfig` (Gemma 4,
   Gemma 1-3, Gemma 3n).
2. Wire mesh inference in `Transformer.init_cache` via the params'
   sharding.
3. Add per-step `with_sharding_constraint` after cache writes in
   each `Attention.__call__`.
4. Add per-step / per-prefill numerical-equivalence + HBM tests.

Estimated diff: ~120 LOC of source + ~200 LOC of tests.

**Phase 3 (optional follow-up).** Address the 2× transient via
prefill donation (§3.3).

---

## 6. Validation / Benchmark Plan

The benchmarks below should be run *before* and *after* each phase to
establish a clean attribution of savings. Use [`examples/cache_probe.py`](./examples/cache_probe.py)
as the primary HBM measurement tool and `time` as decode-rate measurement.

**Hardware:** v5e-4 (4 chips, 16 GB HBM each). v5e-8 if accessible
for 31B confirmation.

**Models:** Gemma4_E2B (sanity, no head-sharding gain), Gemma4_E4B
(primary), Gemma4_31B (extrapolation; v5e-8).

**Workload:** single-turn `chat()` with a fixed prompt and
`max_new_tokens=64`, `multi_turn=False`, `cache_length ∈
{4096, 16384, 32768}`. RNG seeded.

**Metrics per (model, cache_length, change):**

| Metric | Source |
|---|---|
| Per-chip peak HBM during chat() | `device.memory_stats()['peak_bytes_in_use']` |
| Per-chip steady-state HBM after chat() | `device.memory_stats()['bytes_in_use']` |
| End-to-end chat() latency (warm) | `time.time()` around `sampler.chat(...)` |
| Logit hash for first 64 sampled tokens | `hashlib.sha256(jnp.asarray(state.predicted_tokens).tobytes())` |

**Acceptance criteria:**

- HBM: Per-chip peak HBM drops by at least 80% of the predicted
  savings in §3.1 / §3.2 tables. Greedy sampled tokens should match
  before/after the change; logits should stay within BF16 tolerance.
- Latency: ≤ 5% regression on E4B at L=4096. Long-context (L=16384+)
  may improve due to less HBM traffic on the masked region.
- No new compile-time blowups (compile time within 2× of baseline).

---

## 7. Why this is a good idea, briefly

The Gemma-4-specific local-sliding pathology is a structural property
of how `_config.init_cache` is written, not a feature; the masking
code already proves the bytes are unread. The cache replication is
similarly an artifact of "no model knows what mesh the user will run
on" — but the model trivially knows once params have been placed. Both
fixes are local, testable, and unlock long-context evaluation of the
larger Gemma 4 variants on commodity TPU slices. Neither requires
changes to the user-facing API surface.

---

## Appendix A — Diagnostic tooling

This proposal ships two read-only scripts (already in
[`examples/`](./examples/)) used to gather the empirical numbers above.
They are not part of the runtime path.

- **[`examples/cache_audit.py`](./examples/cache_audit.py)** — inspects
  the cache pytree shape/dtype/sharding per leaf without any param
  load. `--shard heads` simulates the proposed Change B sharding so
  reviewers can see what the post-change layout looks like.
- **[`examples/cache_probe.py`](./examples/cache_probe.py)** — runs a
  real `ChatSampler.chat()` with FSDP-sharded params and reports
  per-chip HBM deltas. Used to derive the runtime numbers in §1.1.

Both scripts are useful regression tools to verify Phase 1 and Phase 2
numerically.
