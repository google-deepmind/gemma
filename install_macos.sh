#!/bin/bash
# macOS Installation Script for Gemma
# Addresses mutex issues on Apple Silicon (M1/M2/M3)
#
# This script provides a working installation that avoids the
# "mutex lock failed: Invalid argument" error caused by multiple
# libprotobuf versions in TensorFlow 2.20+
#
# Usage:
#   bash install_macos.sh
#
# Or with conda:
#   bash install_macos.sh --conda

set -e

echo "🍎 Gemma macOS Installation Script"
echo "=================================="
echo ""

# Detect Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Detected Python: $PYTHON_VERSION"

# Check for Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo "Platform: Apple Silicon (M1/M2/M3)"
    APPLE_SILICON=true
else
    echo "Platform: Intel Mac"
    APPLE_SILICON=false
fi
echo ""

# Parse command line arguments
USE_CONDA=false
if [[ "$1" == "--conda" ]]; then
    USE_CONDA=true
fi

# Create conda environment if requested
if [[ "$USE_CONDA" == true ]]; then
    echo "📦 Creating conda environment 'gemma-mac'..."
    conda create -n gemma-mac python=3.11 -y
    echo "Activate with: conda activate gemma-mac"
    echo ""
    echo "Next steps:"
    echo "1. conda activate gemma-mac"
    echo "2. bash $0  # Run this script again without --conda flag"
    exit 0
fi

# Verify we're in a virtual environment
if [[ -z "${VIRTUAL_ENV}" ]] && [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "   Consider using: python3 -m venv venv && source venv/bin/activate"
    echo "   Or: bash $0 --conda"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📥 Installing dependencies..."

# Install NumPy first
echo "Installing NumPy..."
pip install numpy

# Install JAX for CPU (Apple Silicon optimized)
echo "Installing JAX..."
pip install jax

# Install TensorFlow with version constraint to avoid mutex issues
echo "Installing TensorFlow (constrained to <2.20 for mutex compatibility)..."
pip install "tensorflow<2.20"

# Install PyArrow with compatible version
echo "Installing PyArrow (constrained to avoid mutex issues)..."
pip install "pyarrow==22.0.0"

# Install Gemma
echo "Installing Gemma..."
pip install gemma

echo ""
echo "✅ Installation complete!"
echo ""
echo "🧪 Testing installation..."
python3 << 'EOF'
try:
    import jax
    print(f"✓ JAX {jax.__version__} - Devices: {jax.devices()}")
    
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
    
    import pyarrow as pa
    print(f"✓ PyArrow {pa.__version__}")
    
    import gemma
    print(f"✓ Gemma {gemma.__version__}")
    
    print("\n🎉 All packages imported successfully!")
    print("\nTo use Gemma:")
    print("  from gemma import gm")
    print("  model = gm.nn.Gemma3_4B()")
    
except Exception as e:
    print(f"\n❌ Error during import test: {e}")
    print("\nIf you see mutex errors, try:")
    print("  pip uninstall tensorflow pyarrow -y")
    print("  pip install tensorflow==2.19.1 pyarrow==22.0.0")
    exit(1)
EOF

echo ""
echo "📚 For more troubleshooting, see:"
echo "   https://github.com/google-deepmind/gemma/blob/main/TROUBLESHOOTING.md"
