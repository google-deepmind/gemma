# Biomedical Multimodal Dataset Preparation for Gemma Fine-tuning

This repository contains tools and scripts for preparing biomedical multimodal datasets (text + images/tables/formulas) for fine-tuning Gemma models.

## Overview

The process involves three main steps:

1. **Preprocessing PDFs**: Extract text, images, tables, and formulas from biomedical PDFs
2. **Creating a Dataset**: Structure the extracted content into a format suitable for Gemma fine-tuning
3. **Fine-tuning Gemma**: Fine-tune a Gemma model on the prepared dataset

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 1. Preprocessing PDFs

The `preprocess_pdfs.py` script extracts content from PDFs and converts it to appropriate formats:

- **Text**: Extracted as Markdown
- **Images**: Extracted as PNG/JPEG files
- **Tables**: Converted to Markdown format
- **Formulas**: Converted to LaTeX format

### Usage

```bash
python preprocess_pdfs.py --input_dir /path/to/pdfs --output_dir /path/to/preprocessed
```

### Output Structure

```
/path/to/preprocessed/
├── document1/
│   ├── document1.md
│   ├── images/
│   │   ├── img_1_1.png
│   │   ├── img_1_2.png
│   │   └── ...
│   ├── tables/
│   │   ├── table_1.md
│   │   ├── table_2.md
│   │   └── ...
│   └── formulas/
│       ├── formula_1_1.tex
│       ├── formula_1_1.png
│       └── ...
└── document2/
    └── ...
```

## 2. Creating a Dataset

The `create_dataset.py` script structures the preprocessed content into a format suitable for Gemma fine-tuning:

### Usage

```bash
python create_dataset.py --input_dir /path/to/preprocessed --output_dir /path/to/dataset
```

### Output Structure

```
/path/to/dataset/
├── train/
│   └── data.json
├── validation/
│   └── data.json
└── images/
    ├── img_1_1.png
    ├── img_1_2.png
    └── ...
```

### Dataset Format

The dataset is structured as a JSON file with the following format:

```json
[
  {
    "input_text": "Text with <start_of_image> tags and LaTeX formulas",
    "output_text": "Expected output for instruction tuning",
    "images": ["path/to/image1.png", "path/to/image2.png"]
  },
  ...
]
```

## 3. Fine-tuning Gemma

The `finetune_gemma.py` script fine-tunes a Gemma model on the prepared dataset:

### Usage

```bash
python finetune_gemma.py --dataset_dir /path/to/dataset --output_dir /path/to/model
```

## Best Practices

### Handling Images

- Use `<start_of_image>` tags in the text to indicate where images should appear
- Ensure images are properly sized (800x800 pixels is recommended)
- Include descriptive alt text or captions for images

### Handling Tables

- Use Markdown table format to preserve structure
- Keep tables simple and well-formatted
- Ensure column headers are clear and descriptive

### Handling Formulas

- Use LaTeX format for mathematical formulas
- Enclose inline formulas with single dollar signs: `$formula$`
- Enclose block formulas with double dollar signs: `$$formula$$`

### Dataset Preparation

- Balance the dataset with a variety of content types
- Ensure high-quality text and image content
- Provide clear instruction-response pairs for fine-tuning
- Split the dataset into training and validation sets (90/10 split is recommended)

## Limitations and Considerations

- Complex tables may not be perfectly preserved in Markdown format
- Formula extraction accuracy depends on the quality of the PDF
- Very large images may need to be resized or split
- Some special characters in LaTeX formulas may require escaping

## Requirements File

A `requirements.txt` file is included with all necessary dependencies.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
