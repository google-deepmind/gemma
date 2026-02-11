# Implement Tokenizer Auto-Download from GCS

## Description
Implements automatic downloading and caching of tokenizer files from GCS (`gs://`) to the local machine. This improves the developer experience by removing the requirement to manually download vocab files from GCS buckets.

## Changes
- **`gemma/gm/utils/_file_cache.py`**:
    - Updated `maybe_get_from_cache` to detect `gs://` paths.
    - If a `gs://` path is provided and the file is not found in the local cache (`~/.gemma/` by default), it automatically downloads the file using `etils.epath`.
    - Automatically creates necessary local directories.
- **`gemma/gm/text/_tokenizer.py`**:
    - Removed redundant TODOs regarding auto-downloading, as the functionality is now natively supported by the file cache utility.

## Verification
- Created a verification script `verify_tokenizer_download.py` that mocks `etils.epath` and GCS access.
- Verified that:
    1.  Local cache hits return the local path immediately.
    2.  Local cache misses for `gs://` paths trigger a download to the cache directory.
    3.  Local cache misses for standard local paths still return the original paths for compatibility.

## How to test
Set `GEMMA_CACHE_DIR` to a temporary directory if you want to avoid contaminating your `~/.gemma` directory, then initialize a `Gemma2Tokenizer` or `Gemma3Tokenizer`. The model should download the `.model` file on the first run.
