# Gemma

[Gemma](https://ai.google.dev/gemma) is a family of open-weights Large Language 
Model (LLM) by [Google DeepMind](https://deepmind.google/), based on Gemini 
research and technology. 

This repository contains an inference implementation and 
examples, based on the [Flax](https://github.com/google/flax) and 
[JAX](https://github.com/google/jax).

### Learn more about Gemma
 - The [Gemma technical report](https://goo.gle/GemmaReport) details the models'
   capabilities. 
 - For tutorials, reference implementations in other ML frameworks, and more, 
   visit https://ai.google.dev/gemma.

## Quick start

### Installation

1. To install Gemma you need to use Python 3.10 or higher. 

2. Install JAX for CPU, GPU or TPU. Follow instructions at [the JAX website](https://jax.readthedocs.io/en/latest/installation.html).

3. Run 

```
python -m venv gemma-demo
. gemma-demo/bin/activate
pip install git+https://github.com/google-deepmind/gemma.git
```

### Downloading the models

The model checkpoints are available through Kaggle at http://kaggle.com/models/google/gemma.
Please be sure the download Flax checkpoints, as well as the tokenizer.

## Examples

 - [`colabs/sampling_tutorial.py`](https://colab.sandbox.google.com/github/google-deepmind/gemma/blob/main/colabs/sampling_tutorial.py) contains a [Colab](http://colab.google) notebook with a sampling example.

 - [`colabs/fine_tuning_tutorial.py`](https://colab.sandbox.google.com/github/google-deepmind/gemma/blob/main/colabs/fine_tuning_tutorial.py) contains a [Colab](http://colab.google) with a basic tutorial on how to fine tune Gemma for a task, such as English to French
translation.

 - [`colabs/gsm8k_eval.py`](https://colab.sandbox.google.com/github/google-deepmind/gemma/blob/main/colabs/gsm8k_eval.py) is a [Colab](http://colab.google) with a reference GSM8K eval implementation.

## System Requirements
Gemma can run on a CPU, GPU and TPU. For GPU, we recommend a 8GB+ RAM on GPU for
the 2B checkpoint and 24GB+ RAM on GPU for the 7B checkpoint.

## Contributing
We are open to bug reports, pull requests (PR), and other contributions. Please see 
[CONTRIBUTING.md](CONTRIBUTING.md) for details on PRs.

## License
Copyright 2024 DeepMind Technologies Limited

This code is licensed under the Apache License, Version 2.0 (the \"License\"); 
you may not use this file except in compliance with the License. You may obtain
a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed 
under the License is distributed on an AS IS BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the 
specific language governing permissions and limitations under the License.

## Disclaimer
This is not an official Google product.
