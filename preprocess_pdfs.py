#!/usr/bin/env python3
"""
Preprocess biomedical PDFs for Gemma fine-tuning.

This script extracts text, images, tables, and formulas from PDFs and converts them
to appropriate formats for Gemma fine-tuning.

Usage:
    python preprocess_pdfs.py --input_dir /path/to/pdfs --output_dir /path/to/output
"""

import argparse
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import camelot
import pytesseract
from pix2tex.cli import LatexOCR

# Initialize LaTeX OCR model for formula extraction
latex_ocr = LatexOCR()

def extract_images(pdf_path: str, output_dir: str) -> Dict[str, str]:
    """
    Extract images from PDF and save them to output directory.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images
        
    Returns:
        Dictionary mapping image IDs to file paths
    """
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    image_map = {}
    doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Generate a unique ID for the image
            image_id = f"img_{page_num+1}_{img_idx+1}"
            image_filename = f"{image_id}.{image_ext}"
            image_path = os.path.join(image_dir, image_filename)
            
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            # Store the mapping
            image_map[image_id] = image_path
    
    return image_map

def extract_tables(pdf_path: str, output_dir: str) -> Dict[str, str]:
    """
    Extract tables from PDF and convert to Markdown format.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted tables
        
    Returns:
        Dictionary mapping table IDs to Markdown content
    """
    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    
    table_map = {}
    
    # Extract tables using Camelot
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        
        for table_idx, table in enumerate(tables):
            # Generate a unique ID for the table
            table_id = f"table_{table_idx+1}"
            
            # Convert table to Markdown
            markdown_table = table.df.to_markdown(index=False)
            
            # Store the mapping
            table_map[table_id] = markdown_table
            
            # Save the table as Markdown
            table_path = os.path.join(table_dir, f"{table_id}.md")
            with open(table_path, "w") as table_file:
                table_file.write(markdown_table)
    except Exception as e:
        print(f"Error extracting tables: {e}")
    
    return table_map

def extract_formulas(pdf_path: str, output_dir: str) -> Dict[str, str]:
    """
    Extract mathematical formulas from PDF and convert to LaTeX.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted formulas
        
    Returns:
        Dictionary mapping formula IDs to LaTeX content
    """
    formula_dir = os.path.join(output_dir, "formulas")
    os.makedirs(formula_dir, exist_ok=True)
    
    formula_map = {}
    doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(doc):
        # Extract page as an image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Use OCR to identify potential formula regions
        # This is a simplified approach - in practice, you'd need more sophisticated detection
        text = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        # Look for potential formula indicators
        formula_indicators = ["=", "+", "-", "∫", "∑", "∏", "√", "α", "β", "γ", "δ"]
        
        potential_formulas = []
        for i, word in enumerate(text["text"]):
            if any(indicator in word for indicator in formula_indicators):
                x, y, w, h = text["left"][i], text["top"][i], text["width"][i], text["height"][i]
                # Expand the region slightly
                x = max(0, x - 20)
                y = max(0, y - 20)
                w = min(pix.width - x, w + 40)
                h = min(pix.height - y, h + 40)
                
                formula_img = img.crop((x, y, x + w, y + h))
                potential_formulas.append(formula_img)
        
        # Process potential formulas
        for idx, formula_img in enumerate(potential_formulas):
            try:
                # Use LaTeX OCR to convert image to LaTeX
                latex = latex_ocr(formula_img)
                
                # Generate a unique ID for the formula
                formula_id = f"formula_{page_num+1}_{idx+1}"
                
                # Store the mapping
                formula_map[formula_id] = latex
                
                # Save the formula as LaTeX
                formula_path = os.path.join(formula_dir, f"{formula_id}.tex")
                with open(formula_path, "w") as formula_file:
                    formula_file.write(latex)
                
                # Also save the formula image for reference
                formula_img_path = os.path.join(formula_dir, f"{formula_id}.png")
                formula_img.save(formula_img_path)
            except Exception as e:
                print(f"Error processing formula: {e}")
    
    return formula_map

def extract_text_with_references(pdf_path: str, image_map: Dict[str, str], 
                                table_map: Dict[str, str], formula_map: Dict[str, str]) -> str:
    """
    Extract text from PDF with references to images, tables, and formulas.
    
    Args:
        pdf_path: Path to the PDF file
        image_map: Dictionary mapping image IDs to file paths
        table_map: Dictionary mapping table IDs to Markdown content
        formula_map: Dictionary mapping formula IDs to LaTeX content
        
    Returns:
        Markdown text with references to images, tables, and formulas
    """
    doc = fitz.open(pdf_path)
    markdown_text = ""
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        
        # Process text and add references to images, tables, and formulas
        # This is a simplified approach - in practice, you'd need more sophisticated detection
        
        # Add page header
        markdown_text += f"## Page {page_num+1}\n\n"
        
        # Add text
        markdown_text += text + "\n\n"
        
        # Add references to images on this page
        for image_id, image_path in image_map.items():
            if f"_{page_num+1}_" in image_id:
                markdown_text += f"<start_of_image>\n\n"
        
        # Add references to tables on this page
        for table_id, table_content in table_map.items():
            markdown_text += f"\n\n{table_content}\n\n"
        
        # Add references to formulas on this page
        for formula_id, formula_content in formula_map.items():
            if f"_{page_num+1}_" in formula_id:
                markdown_text += f"\n\n${formula_content}$\n\n"
    
    return markdown_text

def preprocess_pdf(pdf_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Preprocess a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save preprocessed content
        
    Returns:
        Dictionary with preprocessed content
    """
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    pdf_output_dir = os.path.join(output_dir, pdf_name)
    os.makedirs(pdf_output_dir, exist_ok=True)
    
    # Extract content
    image_map = extract_images(pdf_path, pdf_output_dir)
    table_map = extract_tables(pdf_path, pdf_output_dir)
    formula_map = extract_formulas(pdf_path, pdf_output_dir)
    
    # Extract text with references
    markdown_text = extract_text_with_references(pdf_path, image_map, table_map, formula_map)
    
    # Save markdown text
    markdown_path = os.path.join(pdf_output_dir, f"{pdf_name}.md")
    with open(markdown_path, "w") as md_file:
        md_file.write(markdown_text)
    
    return {
        "pdf_path": pdf_path,
        "markdown_path": markdown_path,
        "image_map": image_map,
        "table_map": table_map,
        "formula_map": formula_map
    }

def main():
    parser = argparse.ArgumentParser(description="Preprocess biomedical PDFs for Gemma fine-tuning")
    parser.add_argument("--input_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", required=True, help="Directory to save preprocessed content")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of PDF files
    pdf_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".pdf")]
    
    # Process each PDF
    results = []
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        result = preprocess_pdf(pdf_path, args.output_dir)
        results.append(result)
    
    print(f"Processed {len(results)} PDF files")

if __name__ == "__main__":
    main()
