import unittest
import os
from config import get_model_identifier, MODEL_MAP

class TestConfig(unittest.TestCase):
    """
    Unit tests for the configuration and model selection logic.
    """

    def test_model_identifiers(self):
        """Verify that all mapped aliases exist in the MODEL_MAP."""
        self.assertIn("1b", MODEL_MAP)
        self.assertIn("4b", MODEL_MAP)
        self.assertIn("small", MODEL_MAP)
        self.assertEqual(MODEL_MAP["small"], "gemma3:4b")

    def test_environment_override(self):
        """Verify that GEMMA_MODEL_SIZE environment variable correctly overrides the model."""
        # Set environment variable
        os.environ["GEMMA_MODEL_SIZE"] = "small"
        
        # Test the detection
        model = get_model_identifier()
        self.assertEqual(model, "gemma3:4b")
        
        # Clean up
        del os.environ["GEMMA_MODEL_SIZE"]

if __name__ == "__main__":
    unittest.main()
