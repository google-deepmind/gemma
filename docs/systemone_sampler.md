# System One Classifier

The `gm.text.SystemOneSampler` provides ultra-low-latency, non-generative classification, intent routing, and decision triage for Gemma models.

Traditional LLM generation uses an autoregressive decoding loop where tokens are decoded sequentially. In contrast, `SystemOneSampler` evaluates structured decisions via single-pass tree-attention prefill directly from vocabulary logits without generating any new tokens.

Implementation approach is similar to NOKIA Applied Research's [AnyJev](https://github.com/nokia-applied-research/AnyJev)

## Decision Primitives

Modeled after Typesafe AI Jev decision primitives, `SystemOneSampler` supports three zero-shot decision types:

| Primitive | Enum | Description | Output |
| :--- | :--- | :--- | :--- |
| **NOUL** | `gm.text.QuestionType.NOUL` | Calibrated binary decision | `True` or `False`, probabilities, confidence |
| **CHOICE** | `gm.text.QuestionType.CHOICE` | Categorical selection among $K$ options | Option text, index, probability distribution, confidence |
| **SCORE** | `gm.text.QuestionType.SCORE` | Ordinal rating scale (e.g. 1 to 5) | Discrete bucket, continuous expected value $\mathbb{E}[S]$, confidence |

## Option-Order Bias & Cyclic Marginalization

In multi-choice categorical questions, models frequently exhibit **position / option-order bias** (e.g. favoring option A or the final option regardless of context). Changing or reversing the option order can alter the model's prediction on over 20% of items.

```python
# Enable cyclic-shift marginalization in single forward pass:
result = sampler.evaluate_choice(
    state="My order arrived with missing parts and a broken cable.",
    question="Which department should handle this request?",
    options=["Refunds", "Hardware Support", "Shipping", "General Inquiry"],
    marginalize=True,
)
print(result.value)            # Order-invariant selected option
print(result.order_flip_rate)  # Fraction of shifts with argmax disagreement
```

### Label-Free Flip Rate Diagnostic

You can quantify how sensitive a question is to option ordering without any labeled data:

```python
# Evaluate original layout and reversed layout (2 branches in 1 pass):
result = sampler.evaluate_choice(
    state="My order arrived with missing parts...",
    question="Which department should handle this request?",
    options=["Refunds", "Hardware Support", "Shipping", "General Inquiry"],
    check_flip_rate=True,
)
print(result.order_flip_rate)  # 0.0 = order-invariant, 1.0 = argmax flipped
```

A high flip rate indicates that cyclic marginalization will significantly boost accuracy on that task.

## Context-Free Null Calibration

LLMs often exhibit token frequency and semantic priors (e.g. favoring "True" over "False" or affirmative words regardless of context).

`SystemOneSampler` eliminates this prior bias via context-free null calibration:

$$z_{\text{cal}} = z_{\text{real}} - z_{\text{null}}$$

Where $z_{\text{null}}$ represents unnormalized logits evaluated against a null state (e.g. `"N/A"`).

You can precompute null priors once at application startup to achieve maximum serving throughput:

```python
sampler.precompute_null_priors(questions)
```

## Confidence & Temperature Calibration

`SystemOneSampler` reports confidence using **normalized Shannon entropy**:

$$H(P) = -\sum_{i=1}^K P_i \log_2(P_i)$$
$$C = 1 - \frac{H(P)}{\log_2(K)}$$

* $C = 1.0$: Deterministic certainty ($P_k = 1$).
* $C = 0.0$: Maximum uncertainty / uniform distribution ($P_k = \frac{1}{K}$).

Normalized entropy effectively ranks and orders items by certainty. However, at default $T = 1.0$, uncalibrated softmax logits can be overconfident in absolute probability terms.

### Post-Hoc Temperature Scaling

To calibrate probabilities and minimize Expected Calibration Error (ECE), fit a single scalar temperature on a small validation set using `gm.text.fit_temperature` or `gm.text.TemperatureScaler`:

```python
# 1. Measure raw Expected Calibration Error (ECE)
raw_ece = gm.text.compute_ece(val_probs, val_labels)

# 2. Fit scalar temperature minimizing negative log-likelihood (NLL)
cal_temp = gm.text.fit_temperature(val_probs, val_labels)

# 3. Supply calibrated temperature to sampler
sampler = gm.text.SystemOneSampler(
    model=model,
    params=params,
    tokenizer=tokenizer,
    default_temperature=cal_temp,
)
```

### Gemma 4

For Gemma 4 models (e.g., `gemma4-e2b-it`), use `Gemma4Tokenizer` and pass `text_only=True` to exclude media encoders during text classification:

```python
from gemma import gm

# Smallest Gemma 4 checkpoint (2B effective parameters)
model = gm.nn.Gemma4_E2B(text_only=True)
params = gm.ckpts.load_params(
    gm.ckpts.CheckpointPath.GEMMA4_E2B_IT,
    text_only=True,
)
tokenizer = gm.text.Gemma4Tokenizer()

sampler = gm.text.SystemOneSampler(
    model=model,
    params=params,
    tokenizer=tokenizer,
)
```

See the complete runnable example in [`examples/systemone_gemma4.py`](https://github.com/google-deepmind/gemma/blob/main/examples/systemone_gemma4.py).
