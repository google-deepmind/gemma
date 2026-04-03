# Gemma 4

Open-source JAX implementation of the Gemma 4 model.

To use the model:

```
from gemma import gm

# Model and parameters
model = gm.nn.Gemma4_E2B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)

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
