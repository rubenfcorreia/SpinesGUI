from __future__ import annotations

import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from suite2p_numpy_compat import (  # noqa: E402
    ensure_suite2p_numpy_pickle_compat,
    load_suite2p_dict,
    load_suite2p_npy,
)

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
NUMPY2_FIXTURE = FIXTURES / 'suite2p_ops_numpy2.npy'
LEGACY_FIXTURE = FIXTURES / 'suite2p_ops_legacy.npy'
REAL_SESSION_PLANE = Path('/home/rubencorreia/data/Repository/ESRC040/2026-07-16_01_ESRC040/suite2p/plane0')
ALIAS_TARGETS = {
    'numpy._core': 'numpy.core',
    'numpy._core.multiarray': 'numpy.core.multiarray',
    'numpy._core.numeric': 'numpy.core.numeric',
}


class Suite2pNumPyCompatTests(unittest.TestCase):
    def _clear_aliases(self) -> dict[str, object]:
        saved = {name: sys.modules.get(name) for name in ALIAS_TARGETS}
        saved['__numpy_core_present__'] = hasattr(np, '_core')
        saved['__numpy_core__'] = getattr(np, '_core', None)
        for name in ALIAS_TARGETS:
            sys.modules.pop(name, None)
        return saved

    def _restore_aliases(self, saved: dict[str, object]) -> None:
        numpy_core_present = bool(saved.pop('__numpy_core_present__'))
        numpy_core = saved.pop('__numpy_core__')
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if numpy_core_present:
            np._core = numpy_core
        elif hasattr(np, '_core'):
            delattr(np, '_core')

    def test_alias_installation_is_idempotent_and_does_not_overwrite_existing_modules(self) -> None:
        saved = self._clear_aliases()
        try:
            real_find_spec = importlib.util.find_spec

            def fake_find_spec(name, *args, **kwargs):
                if name in ALIAS_TARGETS:
                    return None
                return real_find_spec(name, *args, **kwargs)

            with patch('suite2p_numpy_compat.importlib.util.find_spec', side_effect=fake_find_spec):
                ensure_suite2p_numpy_pickle_compat()
                for alias, target in ALIAS_TARGETS.items():
                    self.assertIn(alias, sys.modules)
                    self.assertIs(sys.modules[alias], importlib.import_module(target))

                sentinels = {name: types.ModuleType(name) for name in ALIAS_TARGETS}
                for name, module in sentinels.items():
                    sys.modules[name] = module

                ensure_suite2p_numpy_pickle_compat()
                for name, module in sentinels.items():
                    self.assertIs(sys.modules[name], module)
        finally:
            self._restore_aliases(saved)

    def test_numpy2_style_suite2p_ops_fixture_loads_through_compatibility_retry(self) -> None:
        saved = self._clear_aliases()
        try:
            self.assertIn(b'numpy._core', NUMPY2_FIXTURE.read_bytes())

            real_load = np.load
            real_find_spec = importlib.util.find_spec
            calls = {'count': 0}

            def flaky_load(*args, **kwargs):
                calls['count'] += 1
                if calls['count'] == 1:
                    raise ModuleNotFoundError("No module named 'numpy._core'")
                return real_load(*args, **kwargs)

            def fake_find_spec(name, *args, **kwargs):
                if name in ALIAS_TARGETS:
                    return None
                return real_find_spec(name, *args, **kwargs)

            with patch('suite2p_numpy_compat.importlib.util.find_spec', side_effect=fake_find_spec):
                with patch('suite2p_numpy_compat.np.load', side_effect=flaky_load):
                    ops = load_suite2p_dict(NUMPY2_FIXTURE)

            self.assertGreaterEqual(calls['count'], 2)
            self.assertEqual(sorted(ops), sorted({
                'Ly', 'Lx', 'data_path', 'datatype', 'frames_per_file',
                'functional_chan', 'max_proj', 'meanImg', 'meanImgE',
                'meanImg_chan2', 'meanImg_chan2_corrected', 'nchannels',
                'nframes', 'reg_file', 'version', 'xrange', 'yrange',
            }))
            self.assertTrue(np.array_equal(ops['meanImg'], np.arange(8, dtype=np.float32).reshape(2, 4)))
            self.assertTrue(np.array_equal(ops['meanImgE'], np.arange(100, 108, dtype=np.float32).reshape(2, 4)))
            self.assertTrue(np.array_equal(ops['meanImg_chan2'], np.arange(300, 308, dtype=np.float32).reshape(2, 4)))
            self.assertTrue(np.array_equal(ops['meanImg_chan2_corrected'], np.arange(400, 408, dtype=np.float32).reshape(2, 4)))
            self.assertTrue(np.array_equal(ops['max_proj'], np.arange(200, 208, dtype=np.float32).reshape(2, 4)))
            self.assertTrue(np.array_equal(ops['yrange'], np.array([1, 3], dtype=np.int64)))
            self.assertTrue(np.array_equal(ops['xrange'], np.array([2, 6], dtype=np.int64)))
            self.assertEqual(ops['Ly'], 8)
            self.assertEqual(ops['Lx'], 8)
            self.assertEqual(ops['nframes'], 123)
            self.assertEqual(ops['datatype'], 'int16')
            self.assertEqual(ops['nchannels'], 2)
            self.assertEqual(ops['functional_chan'], 1)
            self.assertEqual(ops['reg_file'], 'data.bin')
            self.assertEqual(ops['version'], '1.1.0')
            self.assertEqual(ops['data_path'], ['session_a', 'session_b'])
            self.assertEqual(ops['frames_per_file'], [61, 62])
        finally:
            self._restore_aliases(saved)

    def test_legacy_suite2p_ops_fixture_still_loads_without_compatibility_aliases(self) -> None:
        saved = self._clear_aliases()
        try:
            fixture_bytes = LEGACY_FIXTURE.read_bytes()
            self.assertIn(b'numpy.core', fixture_bytes)
            self.assertNotIn(b'numpy._core', fixture_bytes)

            ops = load_suite2p_dict(LEGACY_FIXTURE)
            self.assertEqual(ops['version'], '0.14.4')
            self.assertTrue(np.array_equal(ops['meanImg'], np.arange(8, dtype=np.float32).reshape(2, 4)))
            self.assertTrue(np.array_equal(ops['max_proj'], np.arange(200, 208, dtype=np.float32).reshape(2, 4)))
            self.assertEqual(ops['data_path'], ['session_a', 'session_b'])
        finally:
            self._restore_aliases(saved)

    def test_object_array_roundtrip_uses_the_shared_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stat_file = tmpdir_path / 'stat.npy'
            stat = np.array([
                {'xpix': np.array([1, 2], dtype=np.int64), 'ypix': np.array([3, 4], dtype=np.int64)},
                {'xpix': np.array([5, 6], dtype=np.int64), 'ypix': np.array([7, 8], dtype=np.int64)},
            ], dtype=object)
            np.save(stat_file, stat, allow_pickle=True)

            loaded = load_suite2p_npy(stat_file)
            self.assertIsInstance(loaded, np.ndarray)
            self.assertEqual(loaded.dtype, object)
            self.assertEqual(loaded.shape, (2,))
            self.assertIsInstance(loaded[0], dict)
            self.assertTrue(np.array_equal(loaded[0]['xpix'], np.array([1, 2], dtype=np.int64)))
            self.assertTrue(np.array_equal(loaded[1]['ypix'], np.array([7, 8], dtype=np.int64)))

    def test_unrelated_loading_errors_propagate_without_compatibility_retry(self) -> None:
        with patch('suite2p_numpy_compat.np.load', side_effect=ValueError('boom')) as mocked_load:
            with patch('suite2p_numpy_compat.ensure_suite2p_numpy_pickle_compat') as mocked_compat:
                with self.assertRaisesRegex(ValueError, 'boom'):
                    load_suite2p_npy('does-not-matter.npy')
        mocked_load.assert_called_once()
        mocked_compat.assert_not_called()

    def test_roundtrip_save_and_reload_remains_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            ops_file = tmpdir_path / 'ops.npy'
            ops = {
                'meanImg': np.arange(4, dtype=np.float32).reshape(2, 2),
                'yrange': np.array([0, 2], dtype=np.int64),
                'xrange': np.array([1, 3], dtype=np.int64),
                'Ly': 2,
                'Lx': 2,
                'version': 'test',
            }
            np.save(ops_file, ops, allow_pickle=True)

            loaded = load_suite2p_dict(ops_file)
            self.assertTrue(np.array_equal(loaded['meanImg'], np.arange(4, dtype=np.float32).reshape(2, 2)))
            self.assertTrue(np.array_equal(loaded['yrange'], np.array([0, 2], dtype=np.int64)))
            self.assertTrue(np.array_equal(loaded['xrange'], np.array([1, 3], dtype=np.int64)))
            self.assertEqual(loaded['Ly'], 2)
            self.assertEqual(loaded['Lx'], 2)
            self.assertEqual(loaded['version'], 'test')

    @unittest.skipUnless(REAL_SESSION_PLANE.exists(), 'real session plane is not available')
    def test_real_session_gui_smoke_loads_plane_data(self) -> None:
        try:
            from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
        except ImportError as exc:  # pragma: no cover - only hit if Qt is unavailable
            self.skipTest(f'PyQt5 is not available: {exc}')

        import SpinesGUI as spines_gui

        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'plane0').symlink_to(REAL_SESSION_PLANE, target_is_directory=True)

            app = QApplication.instance() or QApplication([])
            self.addCleanup(app.quit)

            with (
                patch.object(spines_gui.MainWindow, 'showFullScreen', return_value=None),
                patch.object(QFileDialog, 'getExistingDirectory', return_value=str(root)),
                patch.object(QMessageBox, 'information', return_value=QMessageBox.Ok),
                patch.object(QMessageBox, 'warning', return_value=QMessageBox.Ok),
                patch.object(QMessageBox, 'critical', return_value=QMessageBox.Ok),
                patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes),
            ):
                window = spines_gui.MainWindow()
                window.load_suite2p_folder()

                self.assertEqual(window.root_folder, str(root))
                self.assertEqual(window.plane_order, [0])
                self.assertIn(0, window.plane_data)
                plane = window.plane_data[0]
                self.assertEqual(plane['folder'], str(root / 'plane0'))
                self.assertEqual(plane['meanImg'].ndim, 2)
                self.assertGreater(plane['meanImg'].size, 0)
                self.assertEqual(plane['nchannels'], 1)
                self.assertIsNotNone(plane['meanImg_chan2'])
                self.assertEqual(plane['meanImg_chan2'].ndim, 2)
                self.assertGreater(plane['meanImg_chan2'].size, 0)
                self.assertEqual(plane['max_proj'].ndim, 2)
                self.assertGreater(plane['max_proj'].size, 0)

                window._set_view_key('ch2_mean')
                self.assertIsNotNone(window.current_meanImg)
                self.assertTrue(np.array_equal(window.current_meanImg, plane['meanImg_chan2']))

                window._set_view_key('combined')
                self.assertIsNotNone(window.current_combined)
                self.assertEqual(len(window.current_combined), 2)
                self.assertTrue(np.array_equal(window.current_combined[0], plane['meanImg']))
                self.assertTrue(np.array_equal(window.current_combined[1], plane['meanImg_chan2']))

                self.assertEqual(window.roi_data, {})
                window.close()


if __name__ == '__main__':
    unittest.main()
