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

### Prerequisites

* **Python 3.11 or higher:** This library requires modern Python components (e.g. Kauldron and JAX). Using an older Python version (**3.10 or lower**) will cause `pip` to install an unrelated "gemma" package from 2020, resulting in `ImportError: cannot import name 'gm'`.

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

# Model and parameters (Gemma 4)
model = gm.nn.Gemma4_E4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E4B_IT)

# Example of multi-turn conversation
sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    multi_turn=True,
)

prompt = """Which of the 2 images do you prefer ?

Image 1: <|image|>
Image 2: <|image|>

Write your answer as a poem."""
out0 = sampler.chat(prompt, images=[image1, image2])

out1 = sampler.chat('What about the other image ?')
```

The same `ChatSampler` API works with all Gemma versions (2, 3, 3n, 4).

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
  * Gemma 4 (Coming soon)
* Other Gemma implementations and doc on the
  [Gemma ecosystem](https://ai.google.dev/gemma/docs)

### Downloading the models

To download the model weights. See
[our documentation](https://gemma-llm.readthedocs.io/en/latest/checkpoints.html).

### System Requirements

Gemma can run on a CPU, GPU and TPU. For GPU, we recommend 8GB+ RAM on GPU for
The 2B checkpoint and 24GB+ RAM on GPU are used for the 7B checkpoint.

### Contributing

We welcome contributions! Please read our [Contributing Guidelines](./CONTRIBUTING.md) before submitting a pull request.

*This is not an official Google product.*
