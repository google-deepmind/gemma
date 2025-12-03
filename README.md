# Gemma

[![Unittests](https://github.com/google-deepmind/gemma/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-deepmind/gemma/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/gemma.svg)](https://badge.fury.io/py/gemma)
[![Documentation Status](https://readthedocs.org/projects/gemma-llm/badge/?version=latest)](https://gemma-llm.readthedocs.io/en/latest/?badge=latest)

[Gemma](https://ai.google.dev/gemma) is a family of open-weights Large Language
Model (LLM) by [Google DeepMind](https://deepmind.google/), based on Gemini
research and technology.

This repository contains the implementation of the
[`gemma`](https://pypi.org/project/gemma/) PyPI package. A
[JAX](https://github.com/jax-ml/jax) library to use and fine-tune Gemma.

For examples and use cases, see our
[documentation](https://gemma-llm.readthedocs.io/). Please
report issues and feedback in
[our GitHub](https://github.com/google-deepmind/gemma/issues).

### Installation

1.  Install JAX for CPU, GPU or TPU. Follow the instructions on
    [the JAX website](https://jax.readthedocs.io/en/latest/installation.html).
1.  Run

    ```sh
    pip install gemma
    ```

### Examples

Here is a minimal example to have a multi-turn, multi-modal conversation with
Gemma:

```python
from gemma import gm

# Model and parameters
model = gm.nn.Gemma3_4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

# Example of multi-turn conversation
sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    multi_turn=True,
)

prompt = """Which of the two images do you prefer?

Image 1: <start_of_image>
Image 2: <start_of_image>

Write your answer as a poem."""
out0 = sampler.chat(prompt, images=[image1, image2])

out1 = sampler.chat('What about the other image ?')
```

Our documentation contains various Colabs and tutorials, including:

* [Sampling](https://gemma-llm.readthedocs.io/en/latest/colab_sampling.html)
* [Multi-modal](https://gemma-llm.readthedocs.io/en/latest/colab_multimodal.html)
* [Fine-tuning](https://gemma-llm.readthedocs.io/en/latest/colab_finetuning.html)
* [LoRA](https://gemma-llm.readthedocs.io/en/latest/colab_lora_sampling.html)
* ...

Additionally, our
[examples/](https://github.com/google-deepmind/gemma/tree/main/examples) folder
contain additional scripts to fine-tune and sample with Gemma.

### Learn more about Gemma

* To use this library: [Gemma documentation](https://gemma-llm.readthedocs.io/)
* Technical reports for metrics and model capabilities:
  * [Gemma 1](https://goo.gle/GemmaReport)
  * [Gemma 2](https://goo.gle/gemma2report)
  * [Gemma 3](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)
* Other Gemma implementations and doc on the
  [Gemma ecosystem](https://ai.google.dev/gemma/docs)

### Downloading the models

To download the model weights. See
[our documentation](https://gemma-llm.readthedocs.io/en/latest/checkpoints.html).

### System Requirements

Gemma can run on a CPU, GPU and TPU. Below are the recommended hardware specifications:

#### GPU Requirements

**Gemma 2B Checkpoint:**
- **Minimum VRAM:** 8GB+
- **Recommended GPUs:**
  - NVIDIA: RTX 3060 (12GB), RTX 4060 (8GB), RTX 3070 (8GB), RTX 4060 Ti (8GB/16GB)
  - AMD: RX 6600 XT (8GB), RX 6700 XT (12GB), RX 7600 (8GB)
  - Professional: NVIDIA T4 (16GB), A4000 (16GB)

**Gemma 7B Checkpoint:**
- **Minimum VRAM:** 24GB+
- **Recommended GPUs:**
  - NVIDIA: RTX 4090 (24GB), RTX 3090 (24GB), RTX 3090 Ti (24GB)
  - AMD: Radeon RX 7900 XTX (24GB), Radeon RX 7900 XT (20GB)
  - Professional: NVIDIA A5000 (24GB), A6000 (48GB), L40 (48GB), H100 (80GB)

**Quantization Benefits:**
- **Full Precision (Float16/BF16):** ~19GB VRAM for 7B models
- **INT8 Quantization:** ~10GB VRAM for 7B models
- **INT4 Quantization:** ~5GB VRAM for 7B models

> **Note:** Using quantization techniques (INT8/INT4) can significantly reduce VRAM requirements, making 7B models accessible on GPUs with 12-16GB VRAM.

### Contributing

We welcome contributions! Please read our [Contributing Guidelines](./CONTRIBUTING.md) before submitting a pull request.

*This is not an official Google product.*
