# Sampling in Gemma

Sampling determines how the Gemma model selects the next token during text generation from the probability distribution over the vocabulary. This guide explains the core concepts from first principles, the available sampling methods in the Gemma library, practical usage, and recommendations based on hands-on experimentation.

## Table of Contents
- [Why Sampling Matters](#why-sampling-matters)
- [Core Concepts from First Principles](#core-concepts-from-first-principles)
- [Available Sampling Methods in the Gemma Library](#available-sampling-methods-in-the-gemma-library)
- [Using Sampling with ChatSampler (Recommended Interface)](#using-sampling-with-chatsampler-recommended-interface)
- [Using Sampling with Sampler (Lower-Level Control)](#using-sampling-with-sampler-lower-level-control)
- [Recommended Configurations](#recommended-configurations)
- [Common Pitfalls](#common-pitfalls)
- [Advanced Usage](#advanced-usage)
- [References](#references)

## Why Sampling Matters

Gemma, like other large language models (e.g., GPT, LLaMA), outputs a probability distribution over all possible next tokens at each generation step.

**Sampling** is the process of choosing *one* token from this distribution.

- Without sampling (greedy decoding): Always pick the most likely token → deterministic but often repetitive and "robotic."
- With controlled randomness: More diverse, creative, and natural outputs.

Think of Gemma as a **probability engine**. Sampling controls its "personality":
- Greedy/low randomness → Calculator (precise, factual)
- Moderate randomness → Philosopher (balanced reasoning)
- High randomness → Storyteller (creative, exploratory)

Sampling doesn't change the model's knowledge—it only changes how that knowledge is expressed.

## Core Concepts

### 1. Greedy Decoding (No Randomness)

```python
next_token = argmax(probabilities)
```

Always selects the highest-probability token.

**Pros:** Deterministic, fast, good for factual tasks.
**Cons:** Repetitive loops (e.g., "AI is AI is AI..."), bland output.

### 2. Temperature Scaling

Applies a softness to the probability distribution:

$$p_i^{\text{new}} = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}$$

Where:
- $p_i^{\text{new}}$ is the new probability for token $i$
- $\text{logit}_i$ is the raw logit value for token $i$
- $T$ is the temperature parameter
- The denominator ($\sum_j$) sums over all tokens $j$ to normalize the probabilities

- **T = 0.0:** Equivalent to greedy (sharpest).
- **T < 1.0:** Sharper distribution → more confident/focused.
- **T = 1.0:** No change (raw probabilities).
- **T > 1.0:** Flatter distribution → more random/creative.

| Temperature | Behavior |
|------------|----------|
| 0.0–0.5 | Very focused/deterministic |
| 0.7–0.9 | Balanced (recommended default) |
| 1.0+ | Creative (risk of incoherence) |

### 3. Top-k Sampling

Restricts sampling to the k most probable tokens, then samples (after temperature).

- Filters out low-probability "garbage" tokens.
- Common k: 40–100.

### 4. Nucleus (Top-p) Sampling

Keeps the smallest set of tokens whose cumulative probability ≥ p.

- **Adaptive:** Small set when model is confident, larger when uncertain.
- Better than top-k for natural text (from Holtzman et al., 2020).
- Common p: 0.9–0.95.

Nucleus sampling is available in the Gemma library as `NucleusSampling`, added in 2025 to address community requests.

### 5. Repetition Penalty

Not built-in natively in the current Gemma library, but you can approximate with `forbidden_tokens` or custom logit biasing if needed (e.g., penalize recently generated tokens).

## Available Sampling Methods in the Gemma Library

The library (`gemma.gm.text`) provides:

- `gm.text.Greedy()`: Deterministic (default in many cases).
- `gm.text.RandomSampling(temperature=1.0)`: Pure temperature-based.
- `gm.text.TopkSampling(k=50, temperature=1.0)`: Fixed top-k.
- `gm.text.NucleusSampling(p=0.9, temperature=1.0)`: Top-p (recommended for most use cases).

## Using Sampling with ChatSampler (Recommended Interface)

`ChatSampler` handles conversation formatting, multi-turn state, and streaming automatically.

```python
from gemma import gm

model = gm.nn.Gemma3_4B()  # Or your chosen variant
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    multi_turn=True,
    print_stream=True,
    sampling=gm.text.NucleusSampling(p=0.92, temperature=0.85),  # Custom default
)

response = sampler.chat("Explain quantum computing like I'm 10.")
print(response)
```

**Override per-call:**

```python
response = sampler.chat(
    "Write a poem about stars.",
    sampling=gm.text.NucleusSampling(p=0.95, temperature=1.2),  # More creative
)
```

## Using Sampling with Sampler (Lower-Level Control)

For more control over prompt formatting and state management, use `Sampler`:

```python
sampler = gm.text.Sampler(
    model=model,
    params=params,
    sampling=gm.text.NucleusSampling(p=0.9, temperature=0.8),
)

prompt = """<start_of_turn>user
Give me a list of inspirational quotes.<end_of_turn>
<start_of_turn>model
"""

out = sampler.sample(prompt, max_new_tokens=1000)
print(out)
```

## Recommended Configurations

| Use Case | Sampling Config | Why |
|----------|----------------|-----|
| Factual/Q&A/Reasoning | `Greedy()` or `NucleusSampling(p=0.7, temperature=0.6)` | Consistent, low hallucination |
| General chat | `NucleusSampling(p=0.9, temperature=0.8–1.0)` | Natural, coherent |
| Creative writing | `NucleusSampling(p=0.95, temperature=1.1–1.2)` | Diverse, imaginative (watch for incoherence) |

## Common Pitfalls

1. **High temperature (>1.5):** Can lead to gibberish or incoherent outputs.
2. **Very low p (<0.5):** Too restrictive, similar to greedy decoding.
3. **Forgetting `multi_turn=True`:** For conversations, set this to maintain context across turns.
4. **Mixing sampling methods:** Each method has different parameter meanings—don't combine them incorrectly.
5. **Default `k=1` in TopkSampling:** The default `k=1` in `TopkSampling` is effectively greedy. Always set `k` explicitly (e.g., `k=50`) when using top-k sampling.

## Advanced Usage

### Custom Sampling per Turn

You can change sampling strategies dynamically:

```python
sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    multi_turn=True,
)

# First turn: factual
response1 = sampler.chat(
    "What is the capital of France?",
    sampling=gm.text.Greedy(),
)

# Second turn: creative
response2 = sampler.chat(
    "Now write a poem about it.",
    sampling=gm.text.NucleusSampling(p=0.95, temperature=1.1),
)
```

### Using Forbidden Tokens

To prevent certain tokens from being generated (useful for repetition control):

```python
sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    sampling=gm.text.NucleusSampling(p=0.9, temperature=0.8),
    forbidden_tokens=["<end_of_turn>"],  # Prevent early termination
)
```

### Seeding for Reproducibility

For reproducible outputs, pass an `rng` parameter:

```python
import jax

rng = jax.random.PRNGKey(42)  # Fixed seed

response = sampler.chat(
    "Tell me a story.",
    sampling=gm.text.NucleusSampling(p=0.9, temperature=0.8),
    rng=rng,
)
```

## References

- Holtzman et al. (2020): "The Curious Case of Neural Text Degeneration" (introduces nucleus sampling).
- Gemma library source and issues (e.g., nucleus sampling addition in issue #296).
- [Official Gemma Sampling Colab](https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/sampling.ipynb)
