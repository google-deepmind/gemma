import psutil

def get_total_ram_gb():
    """
    Detects the total system RAM and returns it in GB.
    """
    ram_bytes = psutil.virtual_memory().total
    return round(ram_bytes / (1024 ** 3))

def get_recommended_model():
    """
    Recommends a Gemma 3 model size based on detected system RAM.
    
    Recommendation Logic:
    - < 8GB: 1b
    - 8GB - 16GB: 4b
    - 16GB - 32GB: 12b
    - > 32GB: 27b
    """
    ram_gb = get_total_ram_gb()
    
    if ram_gb < 8:
        return '1b'
    elif 8 <= ram_gb < 16:
        return '4b'
    elif 16 <= ram_gb < 32:
        return '12b'
    else:
        return '27b'
