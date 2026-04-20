# macOS Quick Install Guide

This guide provides specific installation instructions for macOS users, particularly those with Apple Silicon (M1/M2/M3).

## Quick Installation (Recommended)

Use the provided installation script:

```bash
# Download and run the script
bash install_macos.sh
```

Or with a new conda environment:

```bash
# Create environment and get instructions
bash install_macos.sh --conda

# Then activate and install
conda activate gemma-mac
bash install_macos.sh
```

## Manual Installation

If you prefer manual installation, follow these steps:

### 1. Create Virtual Environment

```bash
# Using conda (recommended for Apple Silicon)
conda create -n gemma-mac python=3.11
conda activate gemma-mac

# Or using venv
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies in Order

```bash
# Install NumPy first
pip install numpy

# Install JAX for CPU
pip install jax

# Install TensorFlow with version constraint (critical for mutex fix)
pip install "tensorflow<2.20"

# Install PyArrow with compatible version
pip install pyarrow==22.0.0

# Install Gemma
pip install gemma
```

### 3. Verify Installation

```python
from gemma import gm

# This should work without mutex errors
model = gm.nn.Gemma3_4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)
```

## Common Issues

### "mutex lock failed: Invalid argument"

**Cause:** Multiple libprotobuf versions from TensorFlow 2.20+

**Solution:** Downgrade TensorFlow:
```bash
pip uninstall tensorflow -y
pip install "tensorflow<2.20"
```

### "[mutex.cc : 452] RAW: Lock blocking"

**Cause:** PyArrow version incompatibility

**Solution:** Use PyArrow 22.0.0:
```bash
pip uninstall pyarrow -y
pip install pyarrow==22.0.0
```

### Import Errors

**Solution:** Reinstall in a clean environment:
```bash
# Remove old environment
conda env remove -n gemma-mac

# Use the install script
bash install_macos.sh --conda
```

## Platform-Specific Notes

### Apple Silicon (M1/M2/M3)
- Use Python 3.11 (not 3.13.7 which has known issues)
- JAX CPU backend is recommended unless you have Metal support configured
- TensorFlow 2.19.x is the last stable version without mutex issues

### Intel Macs
- Same installation process works
- GPU support not available (JAX CPU only)

## Performance Tips

1. **Use Metal acceleration (Apple Silicon only):**
   ```bash
   pip install tensorflow-metal
   ```

2. **Check JAX device:**
   ```python
   import jax
   print(jax.devices())  # Should show CPU device
   ```

3. **Monitor memory usage:**
   - 2B models: ~8GB RAM minimum
   - 7B models: ~24GB RAM recommended (consider quantization)

## Getting Help

For additional troubleshooting:
- [Full Troubleshooting Guide](./TROUBLESHOOTING.md)
- [GitHub Issues](https://github.com/google-deepmind/gemma/issues)
- [Gemma Documentation](https://gemma-llm.readthedocs.io/)

## Related Issues

- [Issue #426](https://github.com/google-deepmind/gemma/issues/426) - Mutex lock failed on Mac M3
- [TensorFlow #98563](https://github.com/tensorflow/tensorflow/issues/98563) - TF 2.20 PyArrow conflicts
- [Protobuf #21686](https://github.com/protocolbuffers/protobuf/issues/21686) - Multiple libprotobuf versions
