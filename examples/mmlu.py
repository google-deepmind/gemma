r"""An example showing how to load a checkpoint and sample from it.

Getting Started with Gemma Sampling:

Prerequisites:

1. Download your Gemma checkpoint: Choose the desired checkpoint and download it.
2. Get the Gemma tokenizer: Download the tokenizer file required for your model.
3. Install Gemma: Follow the straightforward instructions in the README to install the Gemma repository.

Ready to Sample!

Here's how to run the sampling.py script:

python mmlu.py --path_checkpoint=${PATH_TO_THE_GEMMA_CHECKPOINT} \
    --path_tokenizer=${PATH_TO_THE_GEMMA_TOKENIZER}
"""

import os
import sys
import re
from absl import flags
from absl import app
from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib

import sentencepiece as spm
import datasets

# Define flags
FLAGS = flags.FLAGS

_PATH_CHECKPOINT = flags.DEFINE_string(
    "path_checkpoint", None, required=True, help="Path to checkpoint."
)
_PATH_TOKENIZER = flags.DEFINE_string(
    "path_tokenizer", None, required=True, help="Path to tokenizer."
)
_TOTAL_GENERATION_STEPS = flags.DEFINE_integer(
    "total_generation_steps", 1024, help="Maximum number of steps to run when decoding."
)
_PREAMBLE = flags.DEFINE_string(
    "preamble",
    "The following question is related to machine learning. Please provide a step by step solution to the following question.",
    help="Preamble for the prompt.",
)
_PROMPT = flags.DEFINE_string(
    "prompt",
    """Q: Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.	
Subject: abstract_algebra	
Choices: [ "0", "1", "2", "3" ]	
A: 1

Q: Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.	
Subject: abstract_algebra	
Choices: [ "True, True", "False, False", "True, False", "False, True" ]	
A: 1

Q: Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.	
Subject: abstract_algebra	
Choices: [ "True, True", "False, False", "True, False", "False, True" ]	
A: 2

Q: Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.	
Subject: abstract_algebra	
Choices: [ "True, True", "False, False", "True, False", "False, True" ]	
A: 0

Q: Find the characteristic of the ring 2Z.	
Subject: abstract_algebra	
Choices: [ "0", "3", "12", "30" ]	
A: 0""",
    help="Prompt for the model.",
)

_CACHE_SIZE = 1024

# Load MMLU dataset
mmlu = datasets.load_dataset("cais/mmlu", "machine_learning", cache_dir='/dc/cais_cache')
mmlu_test = mmlu['test']

def _load_and_infer(
    *,
    path_checkpoint: str,
    path_tokenizer: str,
    preamble: str,
    prompt: str,
    total_generation_steps: int,
    cache_size: int,
) -> None:
    """Loads and infers a string from a checkpoint."""
    print(f"Loading the parameters from {path_checkpoint}")
    parameters = params_lib.load_and_format_params(path_checkpoint)
    print("Parameters loaded.")
    
    # Create a sampler with the right param shapes.
    vocab = spm.SentencePieceProcessor()
    vocab.Load(path_tokenizer)
    transformer_config = transformer_lib.TransformerConfig.from_params(
        parameters,
        cache_size=cache_size
    )
    transformer = transformer_lib.Transformer(transformer_config)
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
        params=parameters["transformer"],
    )

    TEMPLATE = """
    Q: {question}
    Subject: {subject}
    Choices: {choices}
    A:"""

    all_correct = 0
    all_responses = {}
    short_responses = {}
    idx = 0
    correct = 0

    for task_id, problem in enumerate(mmlu_test):

        if task_id in all_responses:
            continue

        # Print Task ID
        print(f"task_id {task_id}")

        # Formulate and print the full prompt
        full_prompt = (preamble + '\n\n' + prompt + '\n' +
                       TEMPLATE.format(question=problem['question'],
                                       subject=problem['subject'],
                                       choices=problem['choices']))
        short_prompt = preamble + '\n' + TEMPLATE.format(question=problem['question'],
                                                         subject=problem['subject'],
                                                         choices=problem['choices'])

        input_batch = [full_prompt]
        response = sampler(input_strings=input_batch, total_generation_steps=total_generation_steps)
        print(response.text)

        all_responses[task_id] = response.text[0].split('\nQ:')[0]
        short_responses[task_id] = all_responses[task_id].strip()
        print(f"Short answer: {short_responses[task_id]}")

        try:
            correct += int(problem['answer']) == int(short_responses[task_id])
        except ValueError:
            correct += problem['answer'] == short_responses[task_id]

        print('-'*40)
        print(f"Ground truth answer {problem['answer']}")
        print(f"Short ground truth answer {problem['answer']}")
        print(f"Correct: {correct} out of {idx+1}")
        print("="*40)
        idx += 1

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    _load_and_infer(
        path_checkpoint=_PATH_CHECKPOINT.value,
        path_tokenizer=_PATH_TOKENIZER.value,
        preamble=_PREAMBLE.value,
        prompt=_PROMPT.value,
        total_generation_steps=_TOTAL_GENERATION_STEPS.value,
        cache_size=_CACHE_SIZE,
    )

if __name__ == "__main__":
    app.run(main)
