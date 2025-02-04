# LoRA

## Standalone

```{eval-rst}
.. include:: colab_lora.ipynb
   :parser: myst_nb.docutils_
```

## In the configs

For an end-to-end example, see
[lora.py](https://github.com/google-deepmind/gemma/tree/main/examples/lora.py) config.

You can use LoRA on any config example with 3 simple changes:

1.  Wrap the model in the `gm.nn.LoRAWrapper`. This will apply model surgery to
    replace all the linear and compatible layers with LoRA layers.

    ```python
    model = gm.nn.LoRAWrapper(
        rank=4,
        model=gm.nn.Gemma2_2B(tokens="batch.input"),
    )
    ```

    Internally, this uses the `gemma.peft` mini-library to perform model
    surgery.

1.  Wrap the init transform in a `gm.ckpts.SkipLoRA`. So that only the initial
    pretrained weights are loaded, but the LoRA weights are kept to their random
    initialization.

    ```python
    init_transform = gm.ckpts.SkipLoRA(
        wrapped=gm.ckpts.LoadCheckpoint(
            path=gm.ckpts.CheckpointPath.GEMMA2_2B_IT,
        ),
    )
    ```

1.  Add a mask to the optimizer, so only the LoRA weights are trained.

    ```python
    optimizer = optax.partial_updates(
        optax.adafactor(learning_rate=1e-3),
        mask=kd.optim.select("lora"),
    )
    ```
