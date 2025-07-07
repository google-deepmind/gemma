def split_params(params: _ParamsDict) -> SplittedParams:
  """Split a nested tree into 2 trees, one with and without 'lora' branches."""
  original_tree = {}
  lora_tree = {}

  def _split_recursive(input_subtree, original_subtree, lora_subtree):
    for key, value in input_subtree.items():
      if isinstance(value, dict):
        if key == 'lora':
          lora_subtree[key] = value
        else:
          original_subtree[key] = {}
          lora_subtree[key] = {}
          _split_recursive(value, original_subtree[key], lora_subtree[key])
      elif key != 'lora':
        original_subtree[key] = value

  _split_recursive(params, original_tree, lora_tree)

  # Remove empty dicts in lora_tree
  def _remove_empty_dicts(tree):
    if not isinstance(tree, dict):
      return tree

    new_tree = {}
    for key, value in tree.items():
      if isinstance(value, dict):
        sub_tree = _remove_empty_dicts(value)
        if sub_tree:  # Only add if subtree is not empty
          new_tree[key] = sub_tree
      else:
        new_tree[key] = value
    return new_tree

  lora_tree = _remove_empty_dicts(lora_tree)

  return SplittedParams(original_tree, lora_tree)
