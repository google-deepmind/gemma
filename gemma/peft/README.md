# Parameter Efficient Fine Tuning (PEFT)

This mini-lib to add LoRA support to Flax linen modules.

## LoRA adapters

Some Flax linen `nn.Modules` are available to wrap existing layers:

*   `LoRADense`: Wrap a `nn.Dense` layer.
*   `LoRAEinsum`: Wrap an `nn.Einsum` layer.

Example of usage inside a Flax module:

```python
class MyModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    layer = peft.LoRADense(
        rank=3,
        wrapped=nn.Dense(10),
    )
    return layer(x)
```

Note that each wrapper has an associated low-level module which only perform
the `x @ A @ B` matrix multiplication. For example `peft.LoRADense` ->
`peft.LoRADenseAdapter`. In this case, the sum with the original output has to
be done manually.

```python
class MyModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    dense = nn.Dense(10)
    lora = peft.LoRADenseAdapter(rank=3)  # Only do `x @ A @ B`
    return dense(x) + lora(x)
```

## Model surgery

The library provides some utils to help with model surgery by replacing modules
by their wrapped version. For example:

```python
def _replace_dense_by_lora(module: nn.Module) -> nn.Module:
  if isinstance(module, nn.Dense):
    return peft.LoRADense(rank=3, wrapped=module)
  else:
    return module

# Within the context, the dense layers are replaced by their LoRA version.
with ModuleInterceptor(_replace_dense_by_lora):
  y = model(x)
```

## Params surgery

For params tree structure manipulation:

*   `peft.split_params`: Split a params nested dict into 2 trees: one only containing
    the original params and one only containing the LoRA params.
*   `peft.merge_params`: Reverse of `split_params`. Merge 2 trees into a single tree.

```python
params = {
    'dense': {
        'kernel': 0,
        'bias': 1,
        'lora': {
            'a': 0,
            'b': 1,
        },
    },
    'other': 0,
}

original, lora = peft.split_params(params)
assert original == {
    'dense': {
        'kernel': 0,
        'bias': 1,
    },
    'other': 0,
}
assert lora == {
    'dense': {
        'lora': {
            'a': 0,
            'b': 1,
        },
    },
}
```

To fuse the LoRA params:

*   `peft.fuse_params`: Fuse the LoRA params into the original params weights.
*   `peft.unfuse_params`: Reverse of `fuse_params`, recover the LoRA params.
