import tempfile
import unittest

from spines_extraction import ExtractionCanceled, run_extraction


class ExtractionCancellationTests(unittest.TestCase):
    def test_extraction_honors_cancellation_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ExtractionCanceled):
                run_extraction(d, "normal", False, cancel_check=lambda: True)


if __name__ == "__main__":
    unittest.main()
