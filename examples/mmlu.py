import os
import sentencepiece as spm
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib
from tqdm import tqdm


# Paths to the model and tokenizer
variant = '7b'
weights_dir = "/dc/gemma_models_7b/"
checkpoint_path = os.path.join(weights_dir, variant)
tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')

# Load the parameters
parameters = params_lib.load_and_format_params(checkpoint_path)

# Load the tokenizer
vocab = spm.SentencePieceProcessor()
vocab.Load(tokenizer_path)

# Create the transformer configuration and model
transformer_config = transformer_lib.TransformerConfig.from_params(parameters)
transformer = transformer_lib.Transformer(transformer_config)

# Create the sampler
sampler = sampler_lib.Sampler(
    transformer=transformer,
    vocab=vocab,
    params=parameters["transformer"],
)

# List of available configurations
configs = [
    'machine_learning',
]

# Evaluate the model on a specific configuration of the MMLU dataset
def evaluate_model_on_config(sampler, config):
    dataset = load_dataset("cais/mmlu", config)
    predictions = []
    references = []

    for example in tqdm(dataset['test'], desc=f"Processing {config}"):
        question = example['question']
        choices = example['choices']
        reference = example['answer']

        # Sample the output
        sampled_str = sampler(
            input_strings=[question],
            total_generation_steps=100  # Adjust as needed
        ).text[0]

        # Find the choice that matches the sampled output
        best_choice = max(choices, key=lambda choice: sampled_str in choice)
        predictions.append(best_choice)
        references.append(reference)

    # Calculate accuracy
    accuracy = accuracy_score(references, predictions)
    return accuracy

# Evaluate the model on all configurations
for config in configs:
    accuracy = evaluate_model_on_config(sampler, config)
    print(f"Model accuracy on {config} configuration: {accuracy:.2f}")