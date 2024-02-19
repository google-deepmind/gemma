# This is a Colab notebook. Consider opening in http://colab.google/ or as
# a notebook in JupyterLab.
# %% [markdown]
# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#
# ---
# %% [markdown]
# # GSM8K evaluation using Gemma
#
# The [GSM8K dataset](https://arxiv.org/pdf/2110.14168.pdf) presents a good evaluation challenge for small models for several reasons:
#
# 1. **Conceptual Simplicity:** While the problems in GSM8K require multi-step reasoning, they primarily involve elementary mathematical concepts and basic arithmetic operations. This makes the dataset accessible to smaller models that may not have the capacity to handle complex mathematical reasoning.
#
# 2. **Linguistic Diversity:** GSM8K emphasizes linguistic diversity, ensuring that problems are not simply variations of the same template. This forces models to generalize their understanding of language and mathematical concepts, rather than relying on superficial pattern matching.
#
# 3. **Moderate Difficulty:** The problems in GSM8K are challenging enough to test the limits of small models without being completely intractable. This allows for meaningful evaluation and comparison of different models and methods within a reasonable difficulty range.
#
# 4. **Natural Language Solutions:** GSM8K provides solutions in natural language, encouraging models to develop verbal analytical skills and produce human-interpretable reasoning steps. This is particularly relevant for smaller models that may struggle with purely symbolic or equation-based solutions.
#
# By focusing on grade-school math concepts and emphasizing linguistic diversity, GSM8K provides a valuable benchmark for evaluating the informal reasoning abilities of smaller language models and identifying areas for improvement.
#
# The 2B Gemma checkpoint achieves a score of 19%, which is a higher result than obtained using [much larger competing checkpoints](https://paperswithcode.com/sota/arithmetic-reasoning-on-gsm8k).
# %% [markdown]
# ## Setup
# %% Installation

#Install the necessary dependencies
!pip install --upgrade pip
!pip install --upgrade "jax[gpu]"
!pip install https://github.com/deepmind/gemma
!pip install datasets
# %% Python imports
import re
from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib

from datasets import load_dataset
import sentencepiece as spm
# %% [markdown]
# ## Load GSM8K dataset
# %%
gsm8k = load_dataset("gsm8k", "main", cache_dir='/tmp')
gsm8k_train, gsm8k_test = gsm8k['train'], gsm8k['test']
# %% Download the checkpoints
#TODO: update once the checkpoint's link will be known
ckpt_path = ''
vocab_path = ''
# %% Testing library

def find_numbers(x: str) -> list[str]:
  """Finds all numbers in a string."""
  # Search for number, possibly negative (hyphen), with thousand separators
  # (comma), and with a decimal point (period inbetween digits).
  numbers = re.compile(
      r'-?[\d,]*\.?\d+',
      re.MULTILINE | re.DOTALL | re.IGNORECASE,
  ).findall(x)
  return numbers


def find_number(x: str,
                answer_delimiter: str = 'The answer is') -> str:
  """Finds the most relevant number in a string."""
  # If model uses the answer delimiter, then select the first number following
  # that format.
  if answer_delimiter in x:
    answer = x.split(answer_delimiter)[-1]
    numbers = find_numbers(answer)
    if numbers:
      return numbers[0]

  # In general, select the last number in the string.
  numbers = find_numbers(x)
  if numbers:
    return numbers[-1]
  return ''


def maybe_remove_comma(x: str) -> str:
  # Example: 5,600 -> 5600
  return x.replace(',', '')
# %% GSM8K Prompts

PREAMBLE = """As an expert problem solver solve step by step the following mathematical questions."""

# The default gsm8k prompt from the CoT paper
# https://arxiv.org/pdf/2201.11903.pdf page 35.

PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8."""


# Extension of the default 8-shot prompt, page 35 in
# https://arxiv.org/pdf/2201.11903.pdf
# The extension is intended to improve performance on
# more complicated gsm8k examples.

EXTRA_3_SHOTS = """As an expert problem solver solve step by step the following mathematical questions.

Q: Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?
A: Here's how to calculate Tina's earnings:

**Regular Time:**
- Hours per shift: 8 hours
- Wage per hour: $18.00
- Regular pay per shift: 8 hours * $18.00/hour = $144.00

**Overtime:**
- Overtime hours per shift: 10 hours - 8 hours = 2 hours
- Overtime pay per hour: $18.00 + ($18.00 / 2) = $27.00
- Overtime pay per shift: 2 hours * $27.00/hour = $54.00

**Total per day:**
- Regular pay + overtime pay: $144.00/shift + $54.00/shift = $198.00/day

**Total for 5 days:**
- 5 days * $198.00/day = $990.00

**Therefore, Tina will make $990.00 in 5 days.** The answer is 990.

Q: Abigail is trying a new recipe for a cold drink. It uses 1/4 of a cup of iced tea and 1 and 1/4 of a cup of lemonade to make one drink. If she fills a pitcher with 18 total cups of this drink, how many cups of lemonade are in the pitcher?
A: ## Ambiguity in the Problem Statement:

There is one main ambiguity in the problem statement:

**Total volume vs. Number of servings:** The statement "18 total cups of this drink" could be interpreted in two ways:
  * **18 cups of the combined volume:** This would mean Abigail used a total of 18 cups of liquid, including both iced tea and lemonade.
  * **18 individual servings:** This would mean Abigail made 18 individual drinks, each containing 1/4 cup of iced tea and 1 1/4 cup of lemonade.

Let us assume the interpretation "18 cups of the combined volume".

## Solution assuming 18 cups of combined volume:

**Step 1: Find the proportion of lemonade in one drink:**

* Lemonade: 1 1/4 cups
* Iced tea: 1/4 cup
* Total: 1 1/4 + 1/4 = 1 1/2 cups
* Lemonade proportion: (1 1/4) / (1 1/2) = 5/6

**Step 2: Calculate the amount of lemonade in the pitcher:**

* Total volume: 18 cups
* Lemonade proportion: 5/6
* Volume of lemonade: 18 * (5/6) = 15 cups

Therefore, there are 15 cups of lemonade in the pitcher. The answer is 15.

Q: A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?
A: Let us solve it using algebra. Let x be the number of people on the ship the monster ate in the first hundred years.

The number of people on the ship eaten in the second hundred years is 2x, and in the third hundred years is 4x.

Therefore, the total number of people eaten over three hundred years is x + 2x + 4x = 847.

Combining like terms, we get 7x = 847.

Dividing both sides by 7, we find x = 121.

Therefore, there were 121 people on the ship the monster ate in the first hundred years. The answer is 121."""
# %% [markdown]
# ## Load and prepare your LLM's checkpoint for use with Flax.
#
# Start by loading the weights of your model.
# %%
# Load parameters
params = params_lib.load_and_format_params(ckpt_path)
# %% [markdown]
# Then load the tokenizer.
# %%
vocab = spm.SentencePieceProcessor()
vocab.Load(vocab_path)
# %% [markdown]
# Finally, build a sampler from the transformer configuration deduced from the checkpoint.
# %%
transformer_config = transformer_lib.TransformerConfig.from_params(
    params, cache_size=1024)
transformer = transformer_lib.Transformer(transformer_config)

# Create a sampler with the right param shapes for the GSM8K prompt below
sampler = sampler_lib.Sampler(
    transformer=transformer,
    vocab=vocab,
    params=params['transformer'],
)
# %% [markdown]
# ## Main Evaluation loop
#
# You should expect a score of 19.86% with the 2B model.
# %%
%%time
all_correct = 0
all_responses = {}
short_responses = {}
idx = 0
correct = 0

TEMPLATE = """
Q: {question}
A:"""

for task_id, problem in enumerate(gsm8k_test):

  if task_id in all_responses: continue

  # Print Task ID
  print(f"task_id {task_id}")

  # Formulate and print the full prompt
  full_prompt = (PREAMBLE +'\n\n' + PROMPT + '\n' +
                 TEMPLATE.format(question=problem['question']))
  short_prompt = PREAMBLE +'\n' + TEMPLATE.format(question=problem['question'])

  input_batch = [full_prompt]
  response = sampler(input_strings=input_batch, total_generation_steps=1024)
  print(response.text)

  all_responses[task_id] = response.text[0].split('\nQ:')[0]
  short_responses[task_id] = maybe_remove_comma(find_number(all_responses[task_id]))
  print(f"Short answer: {short_responses[task_id]}")
  try:
    correct += float(maybe_remove_comma(
        find_number(problem['answer']))) == float(short_responses[task_id])
  except:
    correct += maybe_remove_comma(
        find_number(problem['answer'])) == maybe_remove_comma(
            find_number(short_responses[task_id]))
  print('-'*40)
  print(f"Ground truth answer {problem['answer']}")
  print(f"Short ground truth answer {find_number(problem['answer'])}")
  print(f"Correct: {correct} out of {idx+1}")
  print("="*40)
  idx += 1
