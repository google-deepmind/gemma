# Gemma 3n

Open-source JAX implementation of the [Gemma 3n model](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/).

Currently, we only support text generation in the JAX version.

To use the model:
```
from gemma import gm

# Model and parameters
model = gm.nn.gemma3n.Gemma3n_E4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3N_E4B_IT)

# Example of multi-turn conversation
sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    multi_turn=True,
)

prompt = """What's the capital of France?"""
out = sampler.chat(prompt)
print(out)
```
