import unittest
from unittest import mock
import dialog
import numpy as np
from gemma.gm.text import _chat_sampler
from gemma.gm.text import _sampler_loop

class MockTokenizer:
    FORMAT = dialog.Format.GEMMA
    def encode(self, text, add_bos=False):
        return [1] * (len(text) // 5 + 1)
    def decode(self, tokens):
        return "mock_response"

class MockModelInfo:
    tokenizer_version = 4

class MockModel:
    INFO = MockModelInfo()

class MockSamplerState:
    def __init__(self, used_cache_length=0):
        self.used_cache_length = used_cache_length

class MockInnerSampler:
    def __init__(self):
        self.tokenizer = MockTokenizer()
    
    def sample(self, prompt_text, *args, **kwargs):
        from gemma.gm.text import _sampler
        # Simulate returning a response of length 10
        state = MockSamplerState(used_cache_length=kwargs.get('last_state').used_cache_length + 10 if kwargs.get('last_state') else 10)
        return _sampler.SamplerOutput(text="mock response", state=state)

class ChatSamplerTest(unittest.TestCase):
    def test_rolling_cache_turn_pruning(self):
        with mock.patch('gemma.gm.text._chat_sampler.ChatSampler._inner_sampler', new_callable=mock.PropertyMock) as mock_inner:
            mock_inner.return_value = MockInnerSampler()
            
            sampler = _chat_sampler.ChatSampler(
                model=MockModel(),
                params=None,
                multi_turn=True,
                cache_length=20,  # Extremely small cache to force eviction
                max_out_length=5
            )
            sampler.tokenizer = MockTokenizer()
            
            # Turn 1
            out1 = sampler.chat("Hello there, this is a long sentence")
            self.assertEqual(len(sampler.turns), 2)
            
            # Turn 2
            out2 = sampler.chat("Another long sentence to fill the cache")
            self.assertEqual(len(sampler.turns), 4)
            
            # Turn 3 - Should trigger eviction
            # 2 turns * (prompt + response) should exceed cache of 20
            out3 = sampler.chat("This one should push the old ones out!")
            
            # Since the cache size is 20 and we reserve max_out_length=5, it should evict older turns to fit.
            # We expect the turns list to be pruned.
            self.assertTrue(len(sampler.turns) <= 4)
            
if __name__ == '__main__':
    unittest.main()
