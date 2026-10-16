import os

# Mapping of model size aliases to official Ollama/Gemma identifiers
MODEL_MAP = {
    "1b": "gemma3:1b",
    "4b": "gemma3:4b",
    "12b": "gemma3:12b",
    "27b": "gemma3:27b",
    "small": "gemma3:4b",
    "medium": "gemma3:12b",
    "large": "gemma3:27b"
}

def get_model_identifier():
    """
    Retrieves the model identifier based on environment variables or defaults.
    Allows user override via GEMMA_MODEL_SIZE.
    """
    # Check for manual override
    manual_size = os.getenv("GEMMA_MODEL_SIZE")
    if manual_size and manual_size.lower() in MODEL_MAP:
        return MODEL_MAP[manual_size.lower()]
    
    # This will be supplemented by the hardware detector in system_utils.py
    return None
