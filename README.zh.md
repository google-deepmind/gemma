# Gemma

[![Unittests](https://github.com/google-deepmind/gemma/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-deepmind/gemma/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/gemma.svg)](https://badge.fury.io/py/gemma)
[![Documentation Status](https://readthedocs.org/projects/gemma-llm/badge/?version=latest)](https://gemma-llm.readthedocs.io/en/latest/?badge=latest)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

[Gemma](https://ai.google.dev/gemma) 是由 [Google DeepMind](https://deepmind.google/) 基于 Gemini 的研究与技术打造的开源权重（open-weights）大语言模型（LLM）系列。

本仓库包含 [`gemma`](https://pypi.org/project/gemma/) PyPI 软件包的实现——这是一个用于运行和微调 Gemma 的 [JAX](https://github.com/jax-ml/jax) 库。

如需查看代码示例与应用场景，请参阅我们的[官方文档](https://gemma-llm.readthedocs.io/)。若有任何问题或反馈，欢迎在 [GitHub Issues](https://github.com/google-deepmind/gemma/issues) 中提交。

### 安装

1.  安装适用于 CPU、GPU 或 TPU 的 JAX。请按照 [JAX 官方网站](https://jax.readthedocs.io/en/latest/installation.html)上的说明进行操作。
1.  运行以下命令安装 `gemma`：

    ```sh
    pip install gemma
    ```

### 示例

以下是一个与 Gemma 进行多轮、多模态对话的最小示例：

```python
from gemma import gm

# 模型与参数（Gemma 4）
model = gm.nn.Gemma4_E4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E4B_IT)

# 多轮对话示例
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

同一个 `ChatSampler` API 适用于所有 Gemma 版本（2、3、3n、4）。

我们的文档中包含丰富的 Colab 笔记本与教程，涵盖：

* [采样推理 (Sampling)](https://gemma-llm.readthedocs.io/en/latest/colab_sampling.html)
* [多模态 (Multi-modal)](https://gemma-llm.readthedocs.io/en/latest/colab_multimodal.html)
* [模型微调 (Fine-tuning)](https://gemma-llm.readthedocs.io/en/latest/colab_finetuning.html)
* [LoRA 微调与采样 (LoRA)](https://gemma-llm.readthedocs.io/en/latest/colab_lora_sampling.html)
* ……

此外，仓库中的 [examples/](https://github.com/google-deepmind/gemma/tree/main/examples) 目录还提供了更多用于微调与采样推理的示例脚本。

### 深入了解 Gemma

* 库使用指南：[Gemma 官方文档](https://gemma-llm.readthedocs.io/)
* 模型评测指标与能力技术报告：
  * [Gemma 1](https://goo.gle/GemmaReport)
  * [Gemma 2](https://goo.gle/gemma2report)
  * [Gemma 3](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)
  * [Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4)
* 更多 Gemma 实现方案与相关文档请参阅 [Gemma 生态系统](https://ai.google.dev/gemma/docs)

### 下载模型权重

如需下载模型权重，请参阅[权重下载文档](https://gemma-llm.readthedocs.io/en/latest/checkpoints.html)。

### 系统要求

Gemma 支持在 CPU、GPU 和 TPU 上运行。对于 GPU 环境，我们建议为 2B 权重配置 8GB+ 显存（GPU RAM），为 7B 权重配置 24GB+ 显存。

### 参与贡献

我们非常欢迎社区贡献！在提交 Pull Request 之前，请先阅读我们的[贡献指南](./CONTRIBUTING.md)。

*本项目并非 Google 官方正式产品（This is not an official Google product）。*

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月17日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
