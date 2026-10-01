# System One Classifier

The `gm.text.SystemOneSampler` provides ultra-low-latency, non-generative classification, intent routing, and decision triage for Gemma models.

Traditional LLM generation uses an autoregressive decoding loop where tokens are decoded sequentially ($O(T)$). In contrast, `SystemOneSampler` evaluates structured decisions via single-pass tree-attention prefill ($O(1)$) directly from vocabulary logits without generating any new tokens.

## Jev Decision Primitives

Modeled after Jev decision primitives, `SystemOneSampler` supports three zero-shot decision types:

| Primitive | Enum | Description | Output |
| :--- | :--- | :--- | :--- |
| **NOUL** | `gm.text.QuestionType.NOUL` | Calibrated binary decision | `True` or `False`, probabilities, confidence |
| **CHOICE** | `gm.text.QuestionType.CHOICE` | Categorical selection among $K$ options | Option text, index, probability distribution, confidence |
| **SCORE** | `gm.text.QuestionType.SCORE` | Ordinal rating scale (e.g. 1 to 5) | Discrete bucket, continuous expected value $\mathbb{E}[S]$, confidence |

### 1. NOUL (Binary Decision)

Evaluates whether a statement holds for a given context. Uses calibrated logits between affirmative (`True`) and negative (`False`) token subspaces:

```python
result = sampler.evaluate_noul(
    state="Customer received broken hardware and demands immediate refund.",
    question="Is the user requesting a monetary refund?",
)
print(result.value)       # True / False
print(result.confidence)  # Normalized Shannon confidence in [0.0, 1.0]
```

### 2. CHOICE (Categorical Routing)

Selects the best option among candidate categories (e.g., ticket routing, intent detection, topic classification):

```python
result = sampler.evaluate_choice(
    state="Payment gateway returning HTTP 504 timeouts across checkout endpoints.",
    question="Which engineering team is responsible for this alert?",
    options=[
        "Database Infrastructure",
        "Frontend Checkout",
        "Network Operations",
        "Security & Compliance",
    ],
)
print(result.value)           # e.g. "Network Operations"
print(result.selected_index)  # e.g. 2
print(result.probabilities)   # {'Database Infrastructure': 0.12, ...}
```

### 3. SCORE (Ordinal Rating)

Evaluates an integer scale (e.g., 1 to 5) and computes both the modal bucket and the continuous expectation $\mathbb{E}[S] = \sum_{s} s \cdot P(S = s)$:

```python
result = sampler.evaluate_score(
    state="Production database at 99% CPU load with blocked query locks.",
    question="Rate the incident severity on an ITIL scale of 1 to 5.",
    score_range=(1, 5),
)
print(result.value)           # e.g. 5
print(result.expected_value)  # e.g. 4.87
```

## Tree-Attention Prefill ($O(1)$)

When evaluating multiple decision questions on the same input context, running each question through an independent forward pass incurs redundant prefix computation and $N$ forward passes.

`SystemOneSampler` uses **tree-attention packing** (`gm.text.build_tree_attention_pack`):

1. **Shared State Prefix**: The common context is encoded at the root of the sequence.
2. **Question Branches**: All $N$ questions are appended into the same packed sequence.
3. **RoPE Position Restarts**: The position IDs of each branch restart immediately after the shared context prefix ($P = [0 \dots L_{\text{state}}-1, L_{\text{state}} \dots L_{\text{state}}+L_{q_1}-1, L_{\text{state}} \dots L_{\text{state}}+L_{q_2}-1]$), preventing sequential positional drift.
4. **2D Block-Diagonal Causal Masking**: The attention mask allows each branch to attend to the full shared state and to itself causally, but completely masks out other branches.

This guarantees:
* **Zero inter-question contamination**: Branch $B_j$ cannot observe branch $B_i$.
* **Bit-level numerical parity**: Branch logits match an isolated forward pass ($\Delta = 0$).
* **Single forward pass**: All $N$ decisions resolve in a single prefill pass.

```python
response = sampler.evaluate_systemone(
    state="User inquiry or system telemetry...",
    questions=[
        gm.text.QuestionSpec(id="urgent", text="Is urgent?", type=gm.text.QuestionType.NOUL),
        gm.text.QuestionSpec(id="team", text="Route to?", type=gm.text.QuestionType.CHOICE, options=["A", "B", "C"]),
        gm.text.QuestionSpec(id="rating", text="Severity 1-5", type=gm.text.QuestionType.SCORE, score_range=(1, 5)),
    ],
)
```

## Option-Order Bias & Cyclic Marginalization

In multi-choice categorical questions ($K \ge 2$), models frequently exhibit **position / option-order bias** (e.g. favoring option A or the final option regardless of context). Changing or reversing the option order can alter the model's prediction on over 20% of items.

While context-free null calibration fixes token frequency biases on binary decisions, it does not neutralize option-order bias. `SystemOneSampler` resolves order bias through **cyclic-shift marginalization**:

1. Options are presented in $K$ cyclic permutations ($s = 0, \dots, K-1$).
2. All $K$ permutations are evaluated in the **same single tree-attention forward pass ($O(1)$)**, sharing the same state prefix.
3. Probabilities across permutations are marginalized using geometric mean (`logmean`):

$$\log P(i) = \frac{1}{K} \sum_{s=0}^{K-1} \log P_s(i) - \text{const}$$

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

## Empirical Guidelines: When Corrections Help

Based on multi-model benchmark ablations:

| Decision Type | Primary Failure Mode | Recommended Correction | Empirical Impact |
| :--- | :--- | :--- | :--- |
| **NOUL (Binary)** | Label / Token frequency bias (e.g. True vs False) | `calibrate=True` (Null-context prior) | Substantial ECE reduction (+0.06 on injection gates); fixes skewed default marginals. |
| **CHOICE (Multi-way)** | Option-order / Position bias (A vs B vs C) | `marginalize=True` (Cyclic shifts) | +7.3% to +8.7% accuracy improvement; cuts order flip rate from 0.23 down to 0.07. |
| **SCORE (Ordinal)** | Asymmetric distribution skew | `calibrate=True` (Null-context prior) | Centers score distributions and improves expected value calibration. |

## Model Support

`SystemOneSampler` is compatible with all Gemma models:

### Gemma 3 (Recommended for Local Evaluation)

Gemma 3 1B (`gemma3-1b-it`) is compact (~1.5 GB weights) and can be loaded and evaluated locally on CPU or consumer GPU:

```python
from gemma import gm

model = gm.nn.Gemma3_1B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)
tokenizer = gm.text.Gemma3Tokenizer()

sampler = gm.text.SystemOneSampler(
    model=model,
    params=params,
    tokenizer=tokenizer,
)
```

See the complete runnable example in [`examples/systemone_gemma3.py`](https://github.com/google-deepmind/gemma/blob/main/examples/systemone_gemma3.py).

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
