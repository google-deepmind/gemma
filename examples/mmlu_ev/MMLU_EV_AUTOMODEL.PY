from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
dataset = load_dataset("cais/mmlu", "all")
test_set = dataset["test"]


modelname = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(modelname)

model = AutoModelForCausalLM.from_pretrained(modelname, torch_dtype=torch.float16, device_map="auto")


def ANS_GEN(q, choices): # ans

    write= f" Question: {q}\nChoices: {', '.join(choices)}"

    inputs = tokenizer(write, return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_length=300)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)#


    for choice in choices:
        if choice in output:
            return choice
    return None


correct = 0
total = 0

for sample in test_set.select(range(100)):  # 100 SAMPLES
    Q = sample["question"]
    choices = sample["choices"]
    original = choices[sample["answer"]]

    modelanswer = ANS_GEN(Q, choices)

    if modelanswer == original:
        correct += 1
    total += 1

accuracy = (correct / total) * 100
print(f" Model Accuracy  {accuracy:.2f}%")
