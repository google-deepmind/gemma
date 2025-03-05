# Gemma

[![Unittests](https://github.com/google-deepmind/gemma/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-deepmind/gemma/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/gemma.svg)](https://badge.fury.io/py/gemma)
[![Documentation Status](https://readthedocs.org/projects/gemma-llm/badge/?version=latest)](https://gemma-llm.readthedocs.io/en/latest/?badge=latest)

[Gemma](https://ai.google.dev/gemma) is a family of open-weights Large Language
Model (LLM) by [Google DeepMind](https://deepmind.google/), based on Gemini
research and technology.

This repository contain the implementation of the
[`gemma`](https://pypi.org/project/gemma/) PyPI package. A
[JAX](https://github.com/jax-ml/jax) library to use and fine-tune Gemma.

For examples and uses-cases, see our
[documentation](https://gemma-llm.readthedocs.io/). Please
report issues and feedback in
[our GitHub](https://github.com/google-deepmind/gemma/issues).

## Learn more about Gemma

* To use this library: https://gemma-llm.readthedocs.io/
* Technical reports for metrics and model capabilities:
  * [Gemma 1](https://goo.gle/GemmaReport)
  * [Gemma 2](https://goo.gle/gemma2report)
* Other Gemma implementations and doc on the Gemma ecosystem: https://ai.google.dev/gemma/docs

## Quick start

### Installation

1.  Install JAX for CPU, GPU or TPU. Follow instructions at
    [the JAX website](https://jax.readthedocs.io/en/latest/installation.html).
1.  Run

    ```sh
    pip install gemma
    ```

### Downloading the models

To download the model weights. See
[our documentation](https://gemma-llm.readthedocs.io/en/latest/checkpoints.html).

## Examples

Our documentation contain various Colabs and tutorial, including:

* [Sampling](https://gemma-llm.readthedocs.io/en/latest/colab_sampling.html)
* [Fine-tuning](https://gemma-llm.readthedocs.io/en/latest/colab_finetuning.html)
* [LoRA](https://gemma-llm.readthedocs.io/en/latest/colab_lora_sampling.html)
* ...

Additionally, our
[examples/](https://github.com/google-deepmind/gemma/tree/main/examples) folder
contain additional scripts to fine-tune and sample with Gemma.

### System Requirements

Gemma can run on a CPU, GPU and TPU. For GPU, we recommend a 8GB+ RAM on GPU for
the 2B checkpoint and 24GB+ RAM on GPU for the 7B checkpoint.

*This is not an official Google product.*
