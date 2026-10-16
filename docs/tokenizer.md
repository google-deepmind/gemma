# Tokenizer

Gemma tokenizers convert text into token IDs that the model can process. Each Gemma version has its own tokenizer with different vocabulary and special tokens.

For a step-by-step tutorial, see the [tokenizer Colab](colab_tokenizer.ipynb).

## Quick start

```python
from gemma import gm

tokenizer = gm.text.Gemma3Tokenizer()

# Encode text to token IDs
tokenizer.encode('Derinkuyu is an underground city.')
# [8636, 979, 78904, 603, 671, 30073, 3413, 235265]

# Split text into token pieces
tokenizer.split('Derinkuyu is an underground city.')
# ['Der', 'ink', 'uyu', ' is', ' an', ' underground', ' city', '.']

# Decode token IDs back to text
tokenizer.decode([8636, 979, 78904, 603, 671, 30073, 3413, 235265])
# 'Derinkuyu is an underground city.'
```

## Tokenizer versions

Each Gemma model family has a corresponding tokenizer:

```python
tokenizer = gm.text.Gemma2Tokenizer()
tokenizer = gm.text.Gemma3Tokenizer()
tokenizer = gm.text.Gemma3nTokenizer()
```

You can also create a tokenizer by version number:

```python
tokenizer = gm.text.Tokenizer.from_version(3)
```

Feature               | Gemma 2 | Gemma 3   | Gemma 3n
--------------------- | ------- | --------- | --------
Vocab size            | 256,000 | 256,000   | 256,000
Image tokens          | No      | Yes       | Yes
Tool tokens           | No      | Yes       | Yes

## Special tokens

Special tokens are accessible via `tokenizer.special_tokens`:

```python
tokenizer = gm.text.Gemma3Tokenizer()
tokenizer.special_tokens.BOS   # 2
tokenizer.special_tokens.EOS   # 1
```

Token              | Gemma 2 | Gemma 3 / 3n | Description
------------------ | ------- | ------------ | -----------
`PAD`              | 0       | 0            | Padding
`EOS`              | 1       | 1            | End of sentence
`BOS`              | 2       | 2            | Begin of sentence
`UNK`              | 3       | 3            | Unknown
`MASK`             | 4       | 4            | Mask
`CUSTOM`           | 7       | 6            | Start of custom token range
`START_OF_TURN`    | 106     | 105          | `<start_of_turn>`
`END_OF_TURN`      | 107     | 106          | `<end_of_turn>`
`START_OF_IMAGE`   | —       | 255999       | `<start_of_image>` (Gemma 3+ only)
`END_OF_IMAGE`     | —       | 256000       | `<end_of_image>` (Gemma 3+ only)

### BOS / EOS

The `<bos>` token should appear once at the beginning of the input. You can add it with `add_bos=True`:

```python
tokenizer.encode('Hello world!', add_bos=True)
# [2, 4521, 2134, 235341]
```

Similarly, `add_eos=True` appends the end-of-sentence token.

### Turn tokens

Instruction-tuned models use `<start_of_turn>` / `<end_of_turn>` to separate user and model turns:

```python
tokenizer.encode("""<start_of_turn>user
Knock knock.<end_of_turn>
<start_of_turn>model
Who's there?<end_of_turn>""")
```

### Image token

In Gemma 3, `<start_of_image>` marks where an image should be inserted in the prompt. The model expands this internally into soft image tokens.

## Custom tokens

All Gemma tokenizers reserve 99 unused token slots (IDs `CUSTOM + 0` through `CUSTOM + 98`) that can be mapped to custom strings:

```python
tokenizer = gm.text.Gemma3Tokenizer(
    custom_tokens={
        0: '<my_custom_tag>',
        17: '<my_other_tag>',
    },
)

tokenizer.encode('<my_other_tag>')  # [24]
tokenizer.decode(tokenizer.special_tokens.CUSTOM + 17)  # '<my_other_tag>'
```

## Important notes

**Whitespace is part of tokens.** The model treats ` hello` (with leading space) and `hello` as different tokens with different IDs:

```python
tokenizer.encode(' hello')  # [25612]
tokenizer.encode('hello')   # [17534]
```

**Avoid trailing spaces.** When preparing prompts for next-token prediction, a trailing space creates an unusual final token that may degrade quality:

```python
tokenizer.split('The capital of France is ')
# ['The', ' capital', ' of', ' France', ' is', ' ']
#                                               ^^^ unusual trailing token
```
