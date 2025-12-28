"""Demonstration of the SimulateQuantizedEinsum axis selection bug.

This script demonstrates that SimulateQuantizedEinsum.__call__ passes
self.wrapped.name (the module name) to get_axis_to_reduce_from_einsum_str
instead of the actual einsum_str, causing the pattern-specific axis
selection logic to never execute.

Context
-------

In `gemma.peft._quantization.SimulateQuantizedEinsum.__call__`, the
quantization code calls::

    kernel = simulate_quantize(
        kernel,
        self.method,
        axis_to_reduce=get_axis_to_reduce_from_einsum_str(
            einsum_str=self.wrapped.name,  # ← BUG: should be einsum_str
        ),
    )

However, `get_axis_to_reduce_from_einsum_str` is written to match on the
**einsum equation string**, e.g. "BTD,NDH->BTNH", "...H,HF->...F", etc.
Passing the module name (typically something like "einsum_0") means the
matcher always falls through to the default case and returns ``None``.

This script demonstrates the bug by showing:
1. What argument is actually passed to get_axis_to_reduce_from_einsum_str
2. How this affects the quantization axis selection
3. The actual quantization behavior differences
4. Why this causes incorrect quantization scaling
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Any

# Ensure we import from the local code, not an installed package
# Add the parent directory (gemma root) to Python path
_script_dir = Path(__file__).parent
_gemma_root = _script_dir.parent
if str(_gemma_root) not in sys.path:
  sys.path.insert(0, str(_gemma_root))

import jax.numpy as jnp
from flax import linen as nn

from gemma.peft import _quantization
from gemma.peft import _quantization_utils

# Reload the module to ensure we're using the latest code
import importlib
importlib.reload(_quantization)

# Print the code location to verify we're using local code
print(f"Using _quantization from: {_quantization.__file__}")
print(f"Expected location: {_gemma_root / 'gemma' / 'peft' / '_quantization.py'}")
print()

# Global variables to capture execution details
_captured_argument: str | None = None
_captured_axis_result: Any = None
_quantization_called: bool = False
_original_function = _quantization.get_axis_to_reduce_from_einsum_str


def spy_get_axis_to_reduce_from_einsum_str(einsum_str: str) -> Any:
  """Spy function that captures the argument and calls the original."""
  global _captured_argument, _captured_axis_result
  print(f"  [INTERNAL] get_axis_to_reduce_from_einsum_str() called with: '{einsum_str}'")
  _captured_argument = einsum_str
  result = _original_function(einsum_str)
  _captured_axis_result = result
  print(f"  [INTERNAL] get_axis_to_reduce_from_einsum_str() returned: {result}")
  return result


# Also spy on simulate_quantize to see what axis it receives
_original_simulate_quantize = _quantization.simulate_quantize
_captured_quantize_axis: Any = None
_captured_quantize_method: Any = None


def spy_simulate_quantize(
    x: Any,
    method: Any,
    axis_to_reduce: Any = None,
) -> Any:
  """Spy function that captures quantization parameters."""
  global _quantization_called, _captured_quantize_axis, _captured_quantize_method
  _quantization_called = True
  _captured_quantize_axis = axis_to_reduce
  _captured_quantize_method = method
  print(f"  [INTERNAL] simulate_quantize() called with:")
  print(f"    - method: {method}")
  print(f"    - axis_to_reduce: {axis_to_reduce}")
  print(f"    - input shape: {x.shape}")
  result = _original_simulate_quantize(x, method, axis_to_reduce)
  print(f"  [INTERNAL] simulate_quantize() output shape: {result.shape}")
  print(f"  [INTERNAL] simulate_quantize() output range: [{jnp.min(result):.4f}, {jnp.max(result):.4f}]")
  return result


def demonstrate_simulate_quantized_einsum_bug() -> None:
  """Demonstrates that SimulateQuantizedEinsum passes the wrong argument.

  This function shows step-by-step what happens during quantization
  and how the bug affects the actual quantization behavior.
  """
  global _captured_argument, _captured_axis_result, _quantization_called
  global _captured_quantize_axis, _captured_quantize_method

  # The einsum equation we'll use - this is one that get_axis_to_reduce_from_einsum_str
  # knows how to handle and should return (1,) for
  einsum_equation = "BTD,NDH->BTNH"
  expected_axis = (1,)

  print("=" * 80)
  print("STEP-BY-STEP DEMONSTRATION: SimulateQuantizedEinsum Axis Selection Bug")
  print("=" * 80)
  print()
  print("SETUP:")
  print(f"  Einsum equation: {einsum_equation}")
  print(f"  This equation means: (Batch, Time, Dim) @ (Num_heads, Dim, Head_dim) -> (Batch, Time, Num_heads, Head_dim)")
  print(f"  Expected axis for quantization: {expected_axis} (reduces over Dim dimension)")
  print(f"  This allows per-(Num_heads, Head_dim) scaling, which is more accurate")
  print()

  # Patch the functions with our spies
  _quantization.get_axis_to_reduce_from_einsum_str = spy_get_axis_to_reduce_from_einsum_str
  _quantization.simulate_quantize = spy_simulate_quantize
  _captured_argument = None
  _quantization_called = False

  try:
    print("STEP 1: Creating the Einsum module")
    print("-" * 80)
    wrapped_einsum = nn.Einsum(
        einsum_str=einsum_equation,
        shape=(4, 8, 16),  # (N=4 heads, D=8 dim, H=16 head_dim)
        name="attention_proj",
    )
    print(f"  Created nn.Einsum with:")
    print(f"    - einsum_str = '{wrapped_einsum.einsum_str}'")
    print(f"    - name = '{wrapped_einsum.name}'")
    print(f"    - kernel shape = {wrapped_einsum.shape}")
    print()

    print("STEP 2: Wrapping with SimulateQuantizedEinsum")
    print("-" * 80)
    quantized_einsum = _quantization.SimulateQuantizedEinsum(
        wrapped=wrapped_einsum,
        method=_quantization_utils.QuantizationMethod.INT4,
    )
    print(f"  Created SimulateQuantizedEinsum wrapper")
    print(f"    - quantization method: INT4")
    print(f"    - This will quantize the kernel weights to 4-bit integers")
    print()

    print("STEP 3: Initializing the module (Flax requirement)")
    print("-" * 80)
    key = jax.random.key(42)
    dummy_input = jnp.ones((2, 10, 8))  # (Batch=2, Time=10, Dim=8)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  This matches the first operand of einsum: (B, T, D)")
    print()

    variables = quantized_einsum.init(key, dummy_input)
    kernel_before_quant = variables['params']['kernel']
    print(f"  Kernel initialized with shape: {kernel_before_quant.shape}")
    print(f"  Kernel value range: [{jnp.min(kernel_before_quant):.4f}, {jnp.max(kernel_before_quant):.4f}]")
    print()

    print("STEP 4: Calling the module (this triggers quantization)")
    print("-" * 80)
    print("  When quantized_einsum.apply() is called, it will:")
    print("    1. Get the kernel parameter")
    print("    2. Call get_axis_to_reduce_from_einsum_str() to determine quantization axis")
    print("    3. Call simulate_quantize() with that axis")
    print("    4. Use the quantized kernel in the einsum operation")
    print()

    _captured_argument = None
    _captured_axis_result = None
    _quantization_called = False
    output = quantized_einsum.apply(variables, dummy_input)

    print()
    print("=" * 80)
    print("ANALYSIS: What Actually Happened")
    print("=" * 80)
    print()

    print("1. ARGUMENT PASSED TO get_axis_to_reduce_from_einsum_str():")
    print(f"   Received: '{_captured_argument}'")
    print(f"   Expected: '{einsum_equation}'")
    print(f"   Module name: '{wrapped_einsum.name}'")
    print()

    if _captured_argument == wrapped_einsum.name or _captured_argument == 'wrapped':
      print("   ❌ BUG CONFIRMED: The function received the module name")
      print("      instead of the einsum equation string!")
      print()
    elif _captured_argument == einsum_equation:
      print("   ✅ CORRECT: The function received the einsum equation string")
      print("      (Bug appears to be fixed)")
      print()
    else:
      print(f"   ⚠️  UNEXPECTED: Received '{_captured_argument}'")
      print()

    print("2. AXIS SELECTION RESULT:")
    print(f"   get_axis_to_reduce_from_einsum_str('{_captured_argument}') returned: {_captured_axis_result}")
    correct_result = _original_function(einsum_equation)
    print(f"   get_axis_to_reduce_from_einsum_str('{einsum_equation}') would return: {correct_result}")
    print()

    if _captured_axis_result is None:
      print("   ❌ PROBLEM: Returned None means no pattern-specific axis was found")
      print("      The quantization will use generic per-channel scaling")
      print()
    elif _captured_axis_result == expected_axis:
      print("   ✅ CORRECT: Returned the expected axis for this einsum pattern")
      print()

    print("3. QUANTIZATION BEHAVIOR:")
    if _quantization_called:
      print(f"   simulate_quantize() was called with axis_to_reduce = {_captured_quantize_axis}")
      print(f"   This axis determines HOW the quantization scales are computed:")
      print()
      if _captured_quantize_axis is None:
        print("   ❌ With axis_to_reduce=None:")
        print("      - Quantization uses generic 'per-channel' scaling")
        print("      - Scales computed along the LAST axis only (axis=-1)")
        print("      - For kernel shape (4, 8, 16), this means per-H scaling")
        print("      - Each of the 16 head_dim values gets its own scale")
        print("      - But this ignores the (N, D) structure of the tensor")
        print()
        print("   ✅ With axis_to_reduce=(1,) (correct for this einsum):")
        print("      - Quantization uses pattern-specific scaling")
        print("      - Scales computed over axis 1 (the D dimension)")
        print("      - This creates per-(N, H) scaling groups")
        print("      - More accurate because it respects the einsum structure")
        print()
      elif _captured_quantize_axis == expected_axis:
        print("   ✅ Using correct axis_to_reduce=(1,):")
        print("      - Quantization scales computed over the D dimension")
        print("      - Creates per-(N, H) scaling groups")
        print("      - This is the intended behavior for this einsum pattern")
        print()
    else:
      print("   ⚠️  simulate_quantize() was not called (unexpected)")
      print()

    print("4. IMPACT OF THE BUG:")
    print()
    if _captured_axis_result is None:
      print("   ❌ CURRENT BEHAVIOR (with bug):")
      print("      - Pattern-specific axis selection is NEVER used")
      print("      - All einsums get generic per-channel scaling")
      print("      - Quantization accuracy is suboptimal")
      print("      - The code in get_axis_to_reduce_from_einsum_str() is dead code")
      print()
      print("   ✅ EXPECTED BEHAVIOR (after fix):")
      print("      - Pattern-specific axes are correctly identified")
      print("      - Quantization scales match the einsum structure")
      print("      - Better quantization accuracy for attention operations")
      print("      - The pattern matching logic actually works")
      print()
    else:
      print("   ✅ The bug appears to be fixed - correct axis is being used")
      print()

    print("5. HOW WE KNOW THIS IS AN ERROR:")
    print()
    print("   Evidence 1: Wrong argument passed")
    print(f"      - Function received '{_captured_argument}' (module name)")
    print(f"      - Should receive '{einsum_equation}' (einsum equation)")
    print()
    print("   Evidence 2: Pattern matching fails")
    print(f"      - get_axis_to_reduce_from_einsum_str('{_captured_argument}') -> {_captured_axis_result}")
    print(f"      - get_axis_to_reduce_from_einsum_str('{einsum_equation}') -> {correct_result}")
    print(f"      - The function has explicit patterns for einsum equations, not module names")
    print()
    print("   Evidence 3: Dead code")
    print("      - The pattern matching logic in get_axis_to_reduce_from_einsum_str()")
    print("        is never executed because it never receives einsum equations")
    print("      - All calls return None, falling back to generic behavior")
    print()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    if _captured_argument != einsum_equation:
      print("❌ BUG CONFIRMED: SimulateQuantizedEinsum passes the wrong argument")
      print("   to get_axis_to_reduce_from_einsum_str(), causing pattern-specific")
      print("   quantization axis selection to never work.")
      print()
      print("   Fix: Change line 192 in gemma/peft/_quantization.py from:")
      print("     einsum_str=self.wrapped.name")
      print("   to:")
      print("     einsum_str=einsum_str")
    else:
      print("✅ The bug appears to be fixed - correct einsum_str is being passed")
    print()

  finally:
    # Restore the original functions
    _quantization.get_axis_to_reduce_from_einsum_str = _original_function
    _quantization.simulate_quantize = _original_simulate_quantize


if __name__ == "__main__":
  # Import jax here to avoid issues if jax is not available
  import jax

  demonstrate_simulate_quantized_einsum_bug()

