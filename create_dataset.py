#!/usr/bin/env python3
"""
Create a dataset for Gemma fine-tuning from preprocessed biomedical PDFs.

This script creates a dataset structure suitable for Gemma fine-tuning from
preprocessed biomedical PDFs.

Usage:
    python create_dataset.py --input_dir /path/to/preprocessed --output_dir /path/to/dataset
"""

import argparse
import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any

def create_training_example(markdown_path: str, image_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Create a training example from a preprocessed markdown file.
    
    Args:
        markdown_path: Path to the markdown file
        image_map: Dictionary mapping image IDs to file paths
        
    Returns:
        Dictionary with training example data
    """
    # Read markdown content
    with open(markdown_path, "r") as md_file:
        markdown_content = md_file.read()
    
    # Split content into sections (simplified approach)
    sections = markdown_content.split("## Page")
    
    # Create examples
    examples = []
    
    for section in sections:
        if not section.strip():
            continue
        
        # Check if section contains image references
        has_images = "<start_of_image>" in section
        
        # Get image paths for this section
        section_images = []
        if has_images:
            # Extract image IDs from the section
            for image_id in image_map:
                if image_id in section:
                    section_images.append(image_map[image_id])
        
        # Create example
        example = {
            "input_text": section.strip(),
            "output_text": "",  # This would be filled with expected output for instruction tuning
            "images": section_images
        }
        
        examples.append(example)
    
    return examples

def create_dataset(input_dir: str, output_dir: str, split_ratio: float = 0.9) -> None:
    """
    Create a dataset for Gemma fine-tuning.
    
    Args:
        input_dir: Directory containing preprocessed content
        output_dir: Directory to save the dataset
        split_ratio: Train/validation split ratio
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Get list of preprocessed directories
    preprocessed_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Process each directory
    all_examples = []
    
    for dir_name in preprocessed_dirs:
        dir_path = os.path.join(input_dir, dir_name)
        
        # Find markdown file
        markdown_files = [f for f in os.listdir(dir_path) if f.endswith(".md")]
        if not markdown_files:
            continue
        
        markdown_path = os.path.join(dir_path, markdown_files[0])
        
        # Find image map
        image_dir = os.path.join(dir_path, "images")
        if os.path.exists(image_dir):
            image_files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
            image_map = {os.path.splitext(f)[0]: os.path.join(image_dir, f) for f in image_files}
        else:
            image_map = {}
        
        # Create examples
        examples = create_training_example(markdown_path, image_map)
        all_examples.extend(examples)
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Split into train and validation sets
    split_idx = int(len(all_examples) * split_ratio)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    # Copy images to dataset directory and update paths
    for example in all_examples:
        for i, img_path in enumerate(example["images"]):
            img_filename = os.path.basename(img_path)
            new_img_path = os.path.join(output_dir, "images", img_filename)
            shutil.copy(img_path, new_img_path)
            example["images"][i] = os.path.relpath(new_img_path, output_dir)
    
    # Save train and validation sets
    with open(os.path.join(output_dir, "train", "data.json"), "w") as f:
        json.dump(train_examples, f, indent=2)
    
    with open(os.path.join(output_dir, "validation", "data.json"), "w") as f:
        json.dump(val_examples, f, indent=2)
    
    print(f"Created dataset with {len(train_examples)} training examples and {len(val_examples)} validation examples")

def main():
    parser = argparse.ArgumentParser(description="Create a dataset for Gemma fine-tuning")
    parser.add_argument("--input_dir", required=True, help="Directory containing preprocessed content")
    parser.add_argument("--output_dir", required=True, help="Directory to save the dataset")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/validation split ratio")
    args = parser.parse_args()
    
    create_dataset(args.input_dir, args.output_dir, args.split_ratio)

if __name__ == "__main__":
    main()
