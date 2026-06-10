# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for CanvasChunker transform."""

from absl.testing import absltest
from gemma.diffusion.hackable_diffusion_adapter.data import data
import numpy as np

EOS = 1234
PAD = -1


def _make_chunker(num_canvases=3, canvas_size=4):
  return data.CanvasChunker(
      num_canvases=num_canvases,
      canvas_size=canvas_size,
      eos_token=EOS,
      pad_token=PAD,
  )


class CanvasChunkerTest(absltest.TestCase):

  def test_exact_fit(self):
    """Response length == num_canvases * canvas_size."""
    chunker = _make_chunker(num_canvases=3, canvas_size=4)
    response = np.arange(10, 22)  # 12 tokens, exactly 3*4
    features = chunker.map({"response": response})

    np.testing.assert_array_equal(
        features["canvas"],
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    )
    np.testing.assert_array_equal(
        features["canvas_id"],
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    )
    np.testing.assert_array_equal(
        features["canvas_mask"],
        [True] * 12,
    )

  def test_short_response(self):
    """Response shorter than one canvas."""
    chunker = _make_chunker(num_canvases=3, canvas_size=4)
    response = np.array([10, 11])
    features = chunker.map({"response": response})

    np.testing.assert_array_equal(
        features["canvas"],
        [10, 11, EOS, EOS, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD],
    )
    np.testing.assert_array_equal(
        features["canvas_id"],
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    )
    np.testing.assert_array_equal(
        features["canvas_mask"],
        [
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    )

  def test_overflow_truncation(self):
    """Response longer than total capacity."""
    chunker = _make_chunker(num_canvases=2, canvas_size=4)
    response = np.arange(10, 30)  # 20 tokens, capacity is 8
    features = chunker.map({"response": response})

    # Only first 8 tokens kept.
    np.testing.assert_array_equal(
        features["canvas"],
        [10, 11, 12, 13, 14, 15, 16, 17],
    )
    np.testing.assert_array_equal(
        features["canvas_id"],
        [0, 0, 0, 0, 1, 1, 1, 1],
    )
    np.testing.assert_array_equal(
        features["canvas_mask"],
        [True] * 8,
    )

  def test_partial_last_canvas(self):
    """Response fills 2.5 canvases (10 tokens into 3 canvases of 4)."""
    chunker = _make_chunker(num_canvases=3, canvas_size=4)
    response = np.arange(100, 110)  # 10 tokens
    features = chunker.map({"response": response})

    np.testing.assert_array_equal(
        features["canvas"],
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, EOS, EOS],
    )
    np.testing.assert_array_equal(
        features["canvas_id"],
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    )
    # All 3 canvases are valid because canvas 2 has real tokens + EOS.
    np.testing.assert_array_equal(
        features["canvas_mask"],
        [True] * 12,
    )


class CopyFieldTest(absltest.TestCase):

  def test_basic_copy(self):
    """CopyField should copy the value to the new key."""
    transform = data.CopyField(src_key="a", dst_key="b")
    features = transform.map({"a": "hello"})
    self.assertEqual(features["a"], "hello")
    self.assertEqual(features["b"], "hello")

  def test_copy_preserves_other_fields(self):
    """Other fields in the dict should be untouched."""
    transform = data.CopyField(src_key="src", dst_key="dst")
    features = transform.map({"src": "val", "other": 42})
    self.assertEqual(features["src"], "val")
    self.assertEqual(features["dst"], "val")
    self.assertEqual(features["other"], 42)

  def test_copy_numpy_array(self):
    """Copying a numpy array should work (shared reference)."""
    transform = data.CopyField(src_key="arr", dst_key="arr_copy")
    arr = np.array([1, 2, 3])
    features = transform.map({"arr": arr})
    np.testing.assert_array_equal(features["arr"], [1, 2, 3])
    np.testing.assert_array_equal(features["arr_copy"], [1, 2, 3])

  def test_copy_string(self):
    """Copying a string should give an independent value."""
    transform = data.CopyField(src_key="text", dst_key="text_copy")
    features = transform.map({"text": "Bonjour le monde"})
    self.assertEqual(features["text_copy"], "Bonjour le monde")


if __name__ == "__main__":
  absltest.main()
