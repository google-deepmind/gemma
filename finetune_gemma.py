#!/usr/bin/env python3
"""
Fine-tune Gemma on a biomedical multimodal dataset.

This script fine-tunes Gemma on a biomedical multimodal dataset with images, tables, and formulas.

Usage:
    python finetune_gemma.py --dataset_dir /path/to/dataset --output_dir /path/to/output
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import jax.numpy as jnp
import numpy as np
from PIL import Image

from gemma import gm
from kauldron import kd
import optax

class BiomedicalDataset(kd.data.Dataset):
    """Dataset for biomedical multimodal data."""
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer: gm.text.Tokenizer,
        max_length: int = 512,
        training: bool = True,
        sampling: bool = False,
    ):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the JSON data file
            image_dir: Directory containing images
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            training: Whether this is a training dataset
            sampling: Whether this is for sampling
        """
        self.data_path = data_path
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.training = training
        self.sampling = sampling
        
        # Load data
        with open(data_path, "r") as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Get text
        input_text = example["input_text"]
        output_text = example["output_text"]
        
        # Get images
        images = []
        for img_path in example["images"]:
            full_img_path = os.path.join(self.image_dir, img_path)
            if os.path.exists(full_img_path):
                img = np.array(Image.open(full_img_path).convert("RGB"))
                # Resize image to 800x800
                img = kd.data.py.resize_image(img, (800, 800))
                images.append(img)
        
        # If no images, add a dummy image to maintain batch structure
        if not images:
            images = [np.zeros((800, 800, 3), dtype=np.uint8)]
        
        # Stack images
        images = np.stack(images, axis=0)
        
        # If sampling, return the raw text and images
        if self.sampling:
            return {
                "prompt": input_text,
                "response": output_text,
                "image": images,
            }
        
        # Create model inputs
        prompt = self.tokenizer.encode(input_text, add_bos=True)
        response = self.tokenizer.encode(output_text)
        
        # Create the model inputs/targets/loss_mask
        seq2seq_fields = gm.data._functional.make_seq2seq_fields(
            prompt=prompt,
            response=response,
        )
        
        # Add padding
        seq2seq_fields = gm.data._functional.pad(
            seq2seq_fields,
            max_length=self.max_length,
            truncate=True,
        )
        
        return {
            "input": seq2seq_fields.input,
            "target": seq2seq_fields.target,
            "loss_mask": seq2seq_fields.target_mask,
            "image": images,
        }

def get_config(dataset_dir: str, output_dir: str):
    """Get the training configuration.
    
    Args:
        dataset_dir: Directory containing the dataset
        output_dir: Directory to save the output
        
    Returns:
        Training configuration
    """
    batch_size = 8  # Smaller batch size due to multimodal data
    max_length = 512
    
    # Initialize tokenizer
    tokenizer = gm.text.Gemma3Tokenizer()
    
    # Create datasets
    train_data_path = os.path.join(dataset_dir, "train", "data.json")
    val_data_path = os.path.join(dataset_dir, "validation", "data.json")
    image_dir = os.path.join(dataset_dir, "images")
    
    train_ds = BiomedicalDataset(
        data_path=train_data_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        training=True,
    )
    
    val_ds = BiomedicalDataset(
        data_path=val_data_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        training=False,
    )
    
    sampling_ds = BiomedicalDataset(
        data_path=val_data_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        training=False,
        sampling=True,
    )
    
    # Create data loaders
    train_loader = kd.data.PyLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    val_loader = kd.data.PyLoader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    sampling_loader = kd.data.PyLoader(
        dataset=sampling_ds,
        batch_size=1,  # For sampling
        shuffle=False,
        num_workers=1,
    )
    
    return kd.train.Trainer(
        seed=42,
        # Dataset
        train_ds=train_loader,
        # Model definition
        model=gm.nn.Gemma3_4B(
            tokens="batch.input",
            images="batch.image",
        ),
        # Load the weights from the pretrained checkpoint
        init_transform=gm.ckpts.LoadCheckpoint(
            path=gm.ckpts.CheckpointPath.GEMMA3_4B_IT,
        ),
        # Training
        num_train_steps=5000,  # Adjust based on dataset size
        train_losses={
            "xentropy": kd.losses.SoftmaxCrossEntropyWithIntLabels(
                logits="preds.logits",
                labels="batch.target",
                mask="batch.loss_mask",
            ),
        },
        train_summaries={
            "image": kd.summaries.ShowImages(images="batch.image", num_images=5),
        },
        optimizer=optax.adafactor(learning_rate=1e-5),  # Lower learning rate for fine-tuning
        checkpointer=kd.ckpts.Checkpointer(
            save_interval_steps=500,
            workdir=output_dir,
        ),
        # Evaluation
        evals={
            "validation": kd.evals.Evaluator(
                run=kd.evals.EveryNSteps(500),
                ds=val_loader,
            ),
            # The sampler evaluator run inference on a few prompts from the
            # validation set.
            "sampling": gm.evals.SamplerEvaluator(
                run=kd.evals.EveryNSteps(500),
                max_new_tokens=100,  # Sampling parameters
                num_batches=3,
                ds=sampling_loader,
                summaries={
                    "image": kd.summaries.ShowImages(
                        images="batch.image", num_images=5
                    ),
                },
            ),
        },
    )

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma on a biomedical multimodal dataset")
    parser.add_argument("--dataset_dir", required=True, help="Directory containing the dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get configuration
    config = get_config(args.dataset_dir, args.output_dir)
    
    # Start training
    print("Starting fine-tuning...")
    kd.main.run(config)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
