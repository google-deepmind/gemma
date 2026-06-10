# Hackable Diffusion Adapter

This package contains the adapter library for fine-tuning using the
Hackable Diffusion library.

## Setup Instructions

### 1. Python Environment
We recommend Python 3.12 and CUDA 13 for this project.

### 2. Installation
First install the gemma package.

**From PyPI (Recommended)**

```bash
pip install gemma
```

**From Source**

```bash
git clone https://github.com/google-deepmind/gemma.git
cd gemma
pip install .
```

Then we additionally require `jax[cuda13]` dependencies that can be installed
via

```bash
pip install -U jax[cuda13]
```

> [!Note]
> We have tested the library with CUDA 13, other versions can cause
> NCCL errors. Don't mix in other CUDA 12 packages such as
> `jax-cuda12-plugin` or `nvidia-nccl-cu12`.

### 3. Data Preparation
Before training, you need to download and prepare the datasets.
Note that running these scripts requires first cloning the source repository.

#### PubMedQA
To prepare the PubMedQA dataset, make sure your Python environment is
activated, then run:

```bash
cd gemma/diffusion/hackable_diffusion_adapter/data/pubmedqa
bash prepare_pubmedqa_dataset.sh
```

#### Sudoku
To prepare the Sudoku dataset, you first need to configure your Kaggle API
credentials:

- Generate your Kaggle access token.

- Set up the token on your machine:

```bash
mkdir -p ~/.kaggle && echo YOUR_KAGGLE_TOKEN > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```

- Make sure your Python environment is activated, then run the preparation
   script:

```bash
cd gemma/diffusion/hackable_diffusion_adapter/data/sudoku
bash prepare_sudoku_dataset.sh
```

### 4. Training
Once your environment is set up and the data is downloaded, you can kick off a
minimal training run with the following commands.

Command line overrides are used to prevent compilation hangs and NCCL errors.

#### PubMedQA

We fine-tune the model with rank 4 LoRA, using 2 canvases of size 128 each.
We use batch size 2, peak learning rate 1e-4 and train for 2000 steps.
We recommend using a machine with compute capacity at least that of 2 A100s.

From the parent dir of the gemma directory.

```bash
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    NCCL_ALGO="Ring" \
    NCCL_PROTO="LL128" \
    NCCL_NVLS_ENABLE="0" \
    NCCL_CUMEM_ENABLE="0" \
    python3 -m kauldron.main \
  --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_pubmedqa.py \
  --cfg.workdir=$(pwd)/xp_dir
```

#### Sudoku (with LoRA)

We fine-tune the model with rank 8 LoRA, using 1 canvas of size 256.
We use batch size 8, peak learning rate 1.5e-4 and train for 2000 steps.
We recommend using a machine with compute capacity at least that of 2 A100s.

From the parent dir of the gemma directory.

```bash
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    NCCL_ALGO="Ring" \
    NCCL_PROTO="LL128" \
    NCCL_NVLS_ENABLE="0" \
    NCCL_CUMEM_ENABLE="0" \
    python3 -m kauldron.main \
  --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku.py \
  --cfg.workdir=$(pwd)/xp_dir
```

#### Sudoku (full weight updates)

We fine-tune the model using full weight updates with 1 canvas of size 256.
We use batch size 8, peak learning rate 1.5e-4 and train for 2000 steps.
We also use Adafactor for optimization to reduce memory usage.
We recommend using a machine with compute capacity at least that of 8 A100s.

From the parent dir of the gemma directory.

```bash
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    NCCL_ALGO="Ring" \
    NCCL_PROTO="LL128" \
    NCCL_NVLS_ENABLE="0" \
    NCCL_CUMEM_ENABLE="0" \
    python3 -m kauldron.main \
  --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku_full.py \
  --cfg.workdir=$(pwd)/xp_dir
```

### 5. Evaluation

Evaluation is run **offline** as a separate step after training. It loads a
saved checkpoint, runs autoregressive (AR) sampling on the eval dataset, and
reports task-specific metrics (e.g., accuracy for Sudoku, BLEU for PubMedQA).

From the parent dir of the gemma directory:

```bash
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    XLA_PYTHON_CLIENT_PREALLOCATE="false" \
    TF_FORCE_GPU_ALLOW_GROWTH="true" \
    python3 -m gemma.diffusion.hackable_diffusion_adapter.eval_main \
    --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku.py \
    --task=sudoku \
    --step=1000 \
    --eval_names=sample_ar_steps64 \
    --cfg.workdir=$(pwd)/xp_dir_sudoku_lora \
    --cfg.eval_ds.batch_size=2 \
    --cfg.aux.eval_num_batches=2 \
    --cfg.aux.num_canvases=2
```

#### Key Flags

| Flag | Description |
|---|---|
| `--cfg` | Path to the training config file (same one used for training). |
| `--task` | Task to evaluate: `sudoku` or `pubmedqa`. Determines which metrics are reported. |
| `--step` | Checkpoint step to evaluate. If omitted, the latest checkpoint is used. |
| `--eval_names` | Comma-separated list of evaluators to run (e.g. `sample_ar_steps64`). If omitted, all evaluators are run. |
| `--cfg.workdir` | Working directory that contains the training checkpoints. |
| `--cfg.eval_ds.batch_size` | Eval batch size (reduce if running out of memory). |
| `--cfg.aux.eval_num_batches` | Number of eval batches to process. Set to a small value for quick sanity checks, or omit to run over the full eval set. |
| `--cfg.aux.num_canvases` | Number of AR canvases to generate per example. |

#### Available Evaluators

The evaluators are generated automatically from the config. The naming
convention is:

- `sample_ar_steps{N}` — AR diffusion sampling with `N` denoising steps
- `sample_ar_steps{N}_early_stopping` — Same as above, but with
  entropy-based early stopping.

#### Checking Eval Results

Eval metrics are written to TensorBoard event files in the working directory.
To view the results, launch TensorBoard pointing at the workdir:

```bash
tensorboard --logdir=$(pwd)/xp_dir_sudoku_lora
```

Metrics for each evaluator appear under the corresponding eval name
(e.g. `sample_ar_steps64`). For Sudoku, key metrics include overall accuracy,
cell accuracy and difficulty-stratified results. For PubMedQA, look for
accuracy and BLEU scores.
