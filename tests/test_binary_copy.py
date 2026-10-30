import tempfile
import unittest
from pathlib import Path
from unittest import mock

from spines_extraction import _copy_if_incomplete


class BinaryCopyTests(unittest.TestCase):
    def test_missing_destination_is_copied(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source.bin"
            destination = root / "plane" / "data.bin"
            destination.parent.mkdir()
            source.write_bytes(b"source-data")
            self.assertTrue(_copy_if_incomplete(str(source), str(destination)))
            self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_matching_size_skips_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source.bin"
            destination = root / "data.bin"
            source.write_bytes(b"original")
            destination.write_bytes(b"changed!")
            self.assertFalse(_copy_if_incomplete(str(source), str(destination)))
            self.assertEqual(destination.read_bytes(), b"changed!")

    def test_smaller_destination_is_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source.bin"
            destination = root / "data.bin"
            source.write_bytes(b"complete-source")
            destination.write_bytes(b"short")
            self.assertTrue(_copy_if_incomplete(str(source), str(destination)))
            self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_larger_destination_is_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source.bin"
            destination = root / "data.bin"
            source.write_bytes(b"short")
            destination.write_bytes(b"larger-destination")
            self.assertTrue(_copy_if_incomplete(str(source), str(destination)))
            self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_failed_copy_keeps_existing_destination_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source.bin"
            destination = root / "data.bin"
            source.write_bytes(b"complete-source")
            destination.write_bytes(b"old")
            with mock.patch("spines_extraction.shutil.copyfileobj", side_effect=OSError("copy failed")):
                with self.assertRaises(OSError):
                    _copy_if_incomplete(str(source), str(destination))
            self.assertEqual(destination.read_bytes(), b"old")
            self.assertEqual(list(root.glob(".data.bin.tmp-*")), [])

    def test_missing_optional_source_is_reported_without_destination(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(FileNotFoundError):
                _copy_if_incomplete(str(root / "missing-data_chan2.bin"), str(root / "data_chan2.bin"))


if __name__ == "__main__":
    unittest.main()
