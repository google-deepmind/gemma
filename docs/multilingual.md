# Multilingual capability matrix

This page summarizes language and task information stated in the public Gemma
model cards. It is a guide to published evidence, not a guarantee of quality for
a particular language, script, task, or model size. The model cards linked below
are the source of truth and may be updated independently of this page.

## Family-level language information

| Model family | Published language statement | Language-specific inventory |
| ------------ | ---------------------------- | --------------------------- |
| Gemma 1 | Described as available in English. | English is named; no per-language capability matrix is published. |
| Gemma 2 | Described as available in English. | English is named; no per-language capability matrix is published. |
| Gemma 3 | Multilingual support in over 140 languages. | The model card does not enumerate those languages or map them to scripts. |
| Gemma 4 | Out-of-the-box support in over 35 languages; pre-training data includes over 140 languages. | The model card does not enumerate those languages or map them to scripts. |

These are family-level statements. They do not establish that every size,
pre-trained checkpoint, or instruction-tuned checkpoint performs equally well
in every language.

## Language and writing-system reference

The table shows representative language/script identifiers to make the scope of
a future per-language inventory clear. Except where noted, these rows are
examples only: the model cards do not explicitly confirm each listed language
for each family. “Not individually listed” must not be interpreted as either
support or lack of support.

| Language | BCP 47 language tag | Script | Gemma 1 | Gemma 2 | Gemma 3 | Gemma 4 |
| -------- | ------------------ | ------ | ------- | ------- | ------- | ------- |
| English | `en` | Latin (`Latn`) | English named in card | English named in card | Not individually listed | Not individually listed |
| Hindi | `hi` | Devanagari (`Deva`) | Not individually listed | Not individually listed | Not individually listed | Not individually listed |
| Spanish | `es` | Latin (`Latn`) | Not individually listed | Not individually listed | Not individually listed | Not individually listed |
| Japanese | `ja` | Han, Hiragana, and Katakana (`Jpan`) | Not individually listed | Not individually listed | Not individually listed | Not individually listed |

The language examples above are not an exhaustive list or a compatibility
claim. Gemma 3's model card states support for over 140 languages. Gemma 4's
model card distinguishes over 35 languages supported out of the box from over
140 languages represented in pre-training data. Neither card provides a named
list here from which these examples can be verified.

## Task compatibility and evaluation

Model-card descriptions mention general text generation, question answering,
and summarization across the Gemma families. This does not constitute a
language-by-task evaluation. Gemma 4 documents speech-to-translated-text
translation for selected audio-capable variants; that does not establish
general text translation quality or coverage for every language pair.

| Task | Published information | Scope and limitation |
| ---- | --------------------- | -------------------- |
| Text generation | Described as a general model capability. | No comprehensive per-language result matrix is provided here. |
| Summarization | Described as a use case in model-card documentation. | No per-language result matrix is provided here. |
| Question answering | Described as a use case in model-card documentation. | No per-language result matrix is provided here. |
| Text translation | No comprehensive per-language or per-model text-translation compatibility matrix is published here. | Evaluate the exact language direction and checkpoint before relying on it. |
| Audio speech translation | Gemma 4 documents speech-to-translated-text for E2B, E4B, and 12B Unified. | Audio is not listed for Gemma 4 26B A4B or 31B. The card does not provide a complete list of supported source/target language pairs. |

The linked cards and technical reports include benchmark results, but the
published results are not a complete, comparable language-by-language score
table for each task and checkpoint. This page therefore does not reproduce
scores as language-specific evidence. Consult the model card and technical
report for the benchmark setup and results that are available:

* [Gemma 1 model card](https://ai.google.dev/gemma/docs/core/model_card_1) and
  [technical report](https://goo.gle/GemmaReport)
* [Gemma 2 model card](https://ai.google.dev/gemma/docs/core/model_card_2) and
  [technical report](https://goo.gle/gemma2report)
* [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3) and
  [technical report](https://arxiv.org/abs/2503.19786)
* [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) and
  [technical report](https://arxiv.org/abs/2607.02770)

## Limitations and maintenance

* A supported-language count does not identify which languages are included,
  the script or dialect coverage, or the quality level for an individual task.
  For Gemma 4, the pre-training language count is not the same as the out-of-box
  supported-language count.
* Results can differ by model size, checkpoint type, prompt, and language
  direction. The family-level statements above do not resolve those differences.
* “Not individually listed” means the linked model card does not provide a
  per-language statement; it is not evidence that a language is unsupported.
* Add a language/model entry or score only when a model card, technical report,
  or reproducible evaluation explicitly supports that claim. Record the
  evaluation dataset, language or direction, checkpoint, and metric with any
  reported result.
