from datasets import load_dataset
from transformers import pipeline
data_set = load_dataset("cais/mmlu", "all") #   load the dataset

Test_set = data_set["test"]

pipe = pipeline("text-generation", model="google/gemma-3-27b-it") # choose the 27 b model


import random

sampales=random.sample(list(Test_set),100) # taking 100 samples form the data set .

'''
example of the dat set :

{
  "question": "What is the embryological origin of the hyoid bone?",
  "choices": ["The first pharyngeal arch", "The first and second pharyngeal arches", "The second pharyngeal arch", "The second and third pharyngeal arches"],
  "answer": "D"
}



'''
for sample in sampales:
  q=sample["question"]
  choices=sample["choices"]
  original = choices[sample["answer"]]

write= f" Question: {q}\nChoices: {', '.join(choices)}"

ans= pipe(write, max_length=100, do_sample=False)[0]["generated_text"]

print(f"Q: {q}")
print(f"Choices: {choices}")
print(f"LLM Response: {ans}")
print(f"Correct Answer: {original}")
correct=0
total=0
if ans== original:
   correct +=1
total += 1
accuracy = (correct / total) * 100

print(f"accuracy: {accuracy: 2f}%")
