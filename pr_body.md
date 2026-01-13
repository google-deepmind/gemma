# Harden Calculator security (#469) and standardize transformer terminology

This PR addresses critical security vulnerabilities and architectural consistency issues in the `gemma` repository.

### Key Changes:

1.  **Security Hardening (gemma #469)**: Replaced unsafe `eval()` in the `Calculator` tool with a robust, AST-based `_SafeEvaluator`.
    *   Strictly whitelists permitted AST nodes and operations.
    *   Adds support for math constants (`pi`, `e`) and 12+ core functions.
    *   Standardizes float formatting (10 decimal places + scientific notation).
    *   Verified with exhaustive tests (8 cases, 37+ assertions).
2.  **Architecture Standardization**: Renames `num_embed` to `vocab_size` across 25+ files (configs, models, tests) to align with industry standards and maintainer TODOs.
3.  **Broad Compatibility**: Refactored legacy `match` statements to `if/elif` blocks to ensure full compatibility with Python 3.8/3.9 environments.
4.  **Cleanup**: Removed stale `Nucleus Sampling` TODO.

### Verification:
- All new and existing tests pass.
- Repository-wide compilation check completed for Python 3.8+ compatibility.
- Exhaustive mathematical edge cases (nesting, division by zero, etc.) verified.

Detailed proof of work and verification logs are available in the attached walkthrough.
