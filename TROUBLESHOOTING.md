# Troubleshooting Guide

This guide provides solutions to common issues encountered when using Gemma.

## Table of Contents

- [Dependency Conflicts](#dependency-conflicts)
- [Platform-Specific Issues](#platform-specific-issues)

## Dependency Conflicts

### Protobuf Version Conflicts

**Issue:** Conflicts between HuggingFace/Transformers (requires protobuf 4.x) and TensorFlow/MediaPipe (requires protobuf 5.x+).

**Error Messages:**
```
TypeError: Descriptors cannot not be created directly.
AttributeError: module 'google.protobuf.descriptor' has no attribute '_internal_create_key'
```

**Solution 1: Pin Compatible Versions (Recommended)**

For most use cases, pin protobuf to a compatible middle-ground version:

```bash
pip install protobuf==4.25.3 transformers==4.50.2 tensorflow==2.15.0
```

**Solution 2: Use Separate Environments**

For complex workflows requiring both HuggingFace and TensorFlow stacks, use separate conda environments:

```bash
# Environment for HuggingFace/Transformers stack
conda create -n gemma-hf python=3.11
conda activate gemma-hf
pip install torch transformers==4.50.2 trl protobuf==4.25.3 gemma

# Separate environment for TensorFlow/MediaPipe stack
conda create -n gemma-tf python=3.11
conda activate gemma-tf
pip install tensorflow protobuf gemma
```

**Solution 3: Use Maintainer-Provided Colab**

Google DeepMind maintainers provide a working Colab notebook with all dependencies pre-configured:
- [Gemma Dependencies Colab](https://colab.sandbox.google.com/gist/Balakrishna-Chennamsetti/dbc97d48f4730c61fe1b1450e97050ee/gemma_dependecies_issue.ipynb)

**Debugging Dependency Conflicts:**

Check currently installed versions:
```bash
pip show protobuf transformers tensorflow
```

Start fresh if conflicts persist:
```bash
pip freeze | grep -v "^-e" | xargs pip uninstall -y
pip install gemma
```

### Docker Configuration for Consistent Dependencies

For deployment scenarios requiring both stacks:

```dockerfile
FROM python:3.11-slim

# Install compatible versions
RUN pip install --no-cache-dir \
    protobuf==4.25.3 \
    torch \
    transformers==4.50.2 \
    tensorflow==2.15.0 \
    gemma \
    trl

COPY . /app
WORKDIR /app
```

## Platform-Specific Issues

### macOS (Apple Silicon - M1/M2/M3)

**Issue:** `mutex lock failed: Invalid argument` error on Mac M3 Pro with Python 3.13.7.

**Root Cause:** Multiple libprotobuf versions loaded from TensorFlow 2.20+ causing mutex conflicts.

**Solution 1: Downgrade TensorFlow (Recommended)**

```bash
pip install "tensorflow<2.20"
```

**Solution 2: Install Compatible PyArrow**

```bash
pip install pyarrow==22.0.0
```

**Solution 3: Use JAX CPU Backend**

If you don't need TensorFlow features, use JAX with CPU backend:

```bash
pip install jax[cpu]
# Avoid installing tensorflow
```

**Platform-Specific Installation:**

For best results on Apple Silicon:

```bash
# Create fresh environment
conda create -n gemma-mac python=3.11
conda activate gemma-mac

# Install dependencies in order
pip install numpy
pip install jax[cpu]
pip install tensorflow==2.19.1  # Pre-2.20 version
pip install gemma
```

### Windows

**Issue:** `grain` package not available on Windows.

**Solution:** Gemma automatically handles this through conditional dependencies. The `grain` package is excluded on Windows platforms via `sys_platform != 'win32'` constraint in `pyproject.toml`.

**Windows Installation:**

```powershell
# Standard installation works on Windows
pip install gemma

# For GPU support, install JAX for CUDA
pip install jax[cuda12]
```

### Linux

**Issue:** CUDA/GPU compatibility issues.

**Solution:** Ensure proper JAX installation for your CUDA version:

```bash
# For CUDA 12
pip install jax[cuda12]

# For CUDA 11
pip install jax[cuda11]

# Check GPU availability
python -c "import jax; print(jax.devices())"
```

## Getting Help

If you encounter issues not covered in this guide:

1. Check existing [GitHub Issues](https://github.com/google-deepmind/gemma/issues)
2. Search the [Discussions](https://github.com/google-deepmind/gemma/discussions)
3. Open a new issue with:
   - Your platform (OS, Python version, GPU)
   - Full error message and stack trace
   - Output of `pip list` showing installed packages
   - Minimal reproduction code

## Related Resources

- [Installation Guide](https://gemma-llm.readthedocs.io/en/latest/installation.html)
- [JAX Installation](https://jax.readthedocs.io/en/latest/installation.html)
- [System Requirements](./README.md#system-requirements)
