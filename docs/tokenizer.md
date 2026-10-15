# Tokenizer

## Conformance vectors

The repository includes a small, versioned set of Gemma 2 tokenizer conformance
vectors in `gemma/conformance/v1/gemma2_tokenizer.json`. The fixture records the
published tokenizer model source and SHA-256, plus exact token IDs for ASCII,
Unicode, whitespace, and BOS/EOS boundary cases.

Run the vectors against the reference implementation with:

```bash
python -m gemma.conformance
```

To validate a local copy of the tokenizer model instead of the published GCS
path:

```bash
python -m gemma.conformance --tokenizer-path /path/to/tokenizer.model
```

The JSON fixture is intended to be portable so downstream implementations can
run the same cases without depending on the Gemma Python package. Token IDs are
exact comparisons; decoded text is compared exactly as UTF-8 text.
