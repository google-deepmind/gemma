#!/usr/bin/env python3
# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Example script for evaluating Gemma models on the MMLU benchmark.

The Massive Multitask Language Understanding (MMLU) benchmark is a comprehensive
test of knowledge across 57 subjects, including STEM, humanities, social sciences,
and more. This script evaluates Gemma models on the MMLU benchmark by:

1. Loading the MMLU dataset for specified subjects
2. Formatting questions and choices into prompts
3. Generating model responses
4. Calculating accuracy metrics

Example usage:
```sh
python examples/mmlu_eval.py \
    --model_path=/path/to/model \
    --tokenizer_path=/path/to/tokenizer \
    --subject=all  # or specify a specific subject
```
"""

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from gemma import model as gemma_model
from gemma import params as gemma_params
from gemma import sampler as gemma_sampler


def load_mmlu_dataset(subject: str, split: str = "test") -> List[Dict]:
    """Load MMLU dataset for a specific subject.

    Args:
        subject: The MMLU subject to load (e.g., "abstract_algebra", "anatomy").
        split: The dataset split to load ("train", "validation", or "test").

    Returns:
        A list of dictionaries containing the dataset examples.
    """
    dataset = load_dataset("cais/mmlu", subject)
    return dataset[split]


def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    """Format MMLU question and choices into a prompt.

    Args:
        question: The MMLU question text.
        choices: List of answer choices.

    Returns:
        A formatted prompt string.
    """
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65 + i)}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def evaluate_mmlu(
    model: torch.nn.Module,
    tokenizer: gemma_sampler.Tokenizer,
    dataset: List[Dict],
    max_length: int = 2048,
    batch_size: int = 1,
) -> Dict[str, float]:
    """Evaluate model on MMLU dataset.

    Args:
        model: The Gemma model to evaluate.
        tokenizer: The tokenizer for the model.
        dataset: The MMLU dataset to evaluate on.
        max_length: Maximum sequence length for generation.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary containing evaluation metrics.
    """
    correct = 0
    total = 0

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        prompts = []
        answers = []

        for item in batch:
            prompt = format_mmlu_prompt(item["question"], item["choices"])
            prompts.append(prompt)
            answers.append(item["answer"])

        # Generate responses
        responses = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt, add_bos=True)
            tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

            with torch.no_grad():
                output = model.generate(
                    tokens,
                    max_length=max_length,
                    temperature=0.0,
                    top_p=1.0,
                )

            response = tokenizer.decode(output[0].tolist())
            responses.append(response)

        # Evaluate responses
        for response, answer in zip(responses, answers):
            # Extract the first letter of the response as the predicted answer
            predicted = response.strip()[0].upper()
            if predicted == answer:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy}


def main():
    """Main function for MMLU evaluation."""
    parser = argparse.ArgumentParser(description="MMLU evaluation script")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="all",
        help="MMLU subject to evaluate on (default: all)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    # Load model and tokenizer
    model_config = gemma_params.get_model_config("gemma-2b")
    model = gemma_model.GemmaForCausalLM(model_config)
    model.load_weights(args.model_path)
    model = model.to("cuda")
    model.eval()

    tokenizer = gemma_sampler.Tokenizer(args.tokenizer_path)

    # Load and evaluate MMLU dataset
    if args.subject == "all":
        subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology",
            "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "management",
            "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
            "moral_scenarios", "nutrition", "philosophy", "prehistory",
            "professional_accounting", "professional_law", "professional_medicine",
            "professional_psychology", "public_relations", "security_studies",
            "sociology", "us_foreign_policy", "virology", "world_religions"
        ]
    else:
        subjects = [args.subject]

    results = {}
    for subject in subjects:
        print(f"\nEvaluating on subject: {subject}")
        dataset = load_mmlu_dataset(subject)
        metrics = evaluate_mmlu(
            model,
            tokenizer,
            dataset,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        results[subject] = metrics
        print(f"Accuracy: {metrics['accuracy']:.4f}")

    # Save results
    output_file = "mmlu_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
