from __future__ import annotations

import inspect
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from suite2p_compat import (
    extraction_wrapper_compat,
    load_suite2p_functions,
    normalize_suite2p_ops,
    roi_stats_compat,
)


def _stat():
    return np.array([
        {"ypix": np.array([1, 1]), "xpix": np.array([1, 2]), "lam": np.ones(2)},
    ], dtype=object)


class Suite2pCompatTests(unittest.TestCase):
    def test_legacy_roi_stats_receives_legacy_keywords(self):
        calls = {}

        def legacy(stats, Ly, Lx, aspect=None, diameter=None, max_overlap=None, do_crop=True):
            calls.update(locals())
            return stats

        result = roi_stats_compat(
            _stat(), 8, 8, aspect=1.5, diameter=10,
            max_overlap=0.75, do_crop=False, roi_stats=legacy,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(calls["aspect"], 1.5)
        self.assertEqual(calls["diameter"], 10)
        self.assertEqual(calls["do_crop"], False)
        self.assertNotIn("do_soma_crop", calls)

    def test_v1_roi_stats_receives_normalized_keywords_without_aspect(self):
        calls = {}

        def v1(stats, Ly, Lx, diameter=None, max_overlap=0.75, do_soma_crop=True):
            calls.update(locals())
            return stats

        result = roi_stats_compat(
            _stat(), 8, 8, aspect=1.5, diameter=10,
            max_overlap=0.75, do_crop=False, roi_stats=v1,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(calls["diameter"], [15.0, 10.0])
        self.assertEqual(calls["do_soma_crop"], False)
        self.assertNotIn("aspect", inspect.signature(v1).parameters)

    def test_v1_diameter_pair_is_preserved(self):
        calls = {}

        def v1(stats, Ly, Lx, diameter=None, max_overlap=0.75, do_soma_crop=True):
            calls["diameter"] = diameter
            return stats

        roi_stats_compat(_stat(), 8, 8, diameter=[12, 14], roi_stats=v1)
        self.assertEqual(calls["diameter"], [12.0, 14.0])

    def test_v1_extraction_wrapper_maps_settings_and_normalizes_four_outputs(self):
        calls = {}

        def v1(stat, f_reg, f_reg_chan2=None, cell_masks=None,
               neuropil_masks=None, settings=None):
            calls["settings"] = settings
            return ("F", "Fneu", "F2", "Fneu2")

        outputs = extraction_wrapper_compat(
            v1, _stat(), "data", None,
            {"batch_size": 64, "neucoeff": 0.6},
        )
        self.assertEqual(outputs[1:], ("F", "Fneu", "F2", "Fneu2"))
        self.assertEqual(outputs[0][0]["xpix"].tolist(), [1, 2])
        self.assertEqual(calls["settings"]["batch_size"], 64)
        self.assertEqual(calls["settings"]["neuropil_coefficient"], 0.6)
        self.assertEqual(calls["settings"]["lam_percentile"], 50.0)

    def test_v1_extraction_wrapper_can_override_lam_percentile(self):
        calls = {}

        def v1(stat, f_reg, f_reg_chan2=None, cell_masks=None,
               neuropil_masks=None, settings=None):
            calls["settings"] = settings
            return ("F", "Fneu", "F2", "Fneu2")

        extraction_wrapper_compat(
            v1, _stat(), "data", None,
            {"extraction": {"lam_percentile": 50.0}},
            lam_percentile_override=0.0,
        )
        self.assertEqual(calls["settings"]["lam_percentile"], 0.0)

    def test_large_radius_override_avoids_percentile_filter_path(self):
        stat = np.array([
            {
                "ypix": np.array([1, 1]),
                "xpix": np.array([1, 2]),
                "lam": np.ones(2),
                "radius": 150.0,
            },
        ], dtype=object)

        def v1(stat, f_reg, f_reg_chan2=None, cell_masks=None,
               neuropil_masks=None, settings=None):
            if np.median([item["radius"] for item in stat]) * 5 > 500:
                if settings["lam_percentile"] > 0:
                    raise MemoryError("percentile filter would be oversized")
            return ("F", "Fneu", "F2", "Fneu2")

        outputs = extraction_wrapper_compat(
            v1, stat, "data", None, {},
            lam_percentile_override=0.0,
        )
        self.assertEqual(outputs[1:], ("F", "Fneu", "F2", "Fneu2"))

    def test_legacy_extraction_override_does_not_mutate_ops(self):
        calls = {}

        def legacy(stat, f_reg, f_reg_chan2=None, cell_masks=None,
                   neuropil_masks=None, ops=None):
            calls["ops"] = ops
            return (stat, "F", "Fneu", None, None)

        original = {"lam_percentile": 50.0, "extraction": {"lam_percentile": 50.0}}
        extraction_wrapper_compat(
            legacy, _stat(), "data", None, original,
            lam_percentile_override=0.0,
        )
        self.assertEqual(calls["ops"]["lam_percentile"], 0.0)
        self.assertEqual(calls["ops"]["extraction"]["lam_percentile"], 0.0)
        self.assertEqual(original["lam_percentile"], 50.0)
        self.assertEqual(original["extraction"]["lam_percentile"], 50.0)

    def test_legacy_extraction_wrapper_keeps_five_output_contract(self):
        calls = {}

        def legacy(stat, f_reg, f_reg_chan2=None, cell_masks=None,
                   neuropil_masks=None, ops=None):
            calls["ops"] = ops
            return (stat, "F", "Fneu", None, None)

        outputs = extraction_wrapper_compat(legacy, _stat(), "data", None, {"x": 1})
        self.assertEqual(len(outputs), 5)
        self.assertEqual(outputs[1], "F")
        self.assertEqual(calls["ops"], {"x": 1})

    def test_nested_v1_and_flat_legacy_configs_normalize_identically(self):
        legacy = normalize_suite2p_ops({
            "Ly": 512, "Lx": 256, "diameter": 10,
            "aspect": 1, "max_overlap": 0.75, "soma_crop": 1,
        })
        v1 = normalize_suite2p_ops({
            "registration": {"Ly": 512, "Lx": 256},
            "detection": {"diameter": [10, 10], "max_overlap": 0.75},
            "extraction": {"do_soma_crop": False},
        })
        self.assertEqual((legacy["Ly"], legacy["Lx"]), (v1["Ly"], v1["Lx"]))
        self.assertEqual(legacy["max_overlap"], v1["max_overlap"])
        self.assertEqual(v1["diameter"], [10, 10])
        self.assertFalse(v1["soma_crop"])

    def test_v1_neuropil_and_deconvolution_defaults_are_flattened(self):
        normalized = normalize_suite2p_ops({
            "neucoeff": None,
            "prctile_baseline": None,
            "extraction": {"neuropil_coefficient": 0.7},
        })
        self.assertEqual(normalized["neucoeff"], 0.7)
        self.assertEqual(normalized["prctile_baseline"], 8)

    def test_loader_prefers_v1_detection_stats_path(self):
        suite2p = types.ModuleType("suite2p")
        suite2p.__file__ = "/fake/suite2p/__init__.py"
        suite2p.__version__ = "1.1.0"
        detection = types.ModuleType("suite2p.detection")
        stats = types.ModuleType("suite2p.detection.stats")
        extraction = types.ModuleType("suite2p.extraction")
        binary = types.ModuleType("suite2p.io.binary")
        io_module = types.ModuleType("suite2p.io")
        dcnv = types.ModuleType("suite2p.extraction.dcnv")
        stats.roi_stats = lambda *args, **kwargs: args[0]
        extraction.extraction_wrapper = lambda *args, **kwargs: None
        binary.BinaryFile = object
        dcnv.oasis = lambda *args, **kwargs: None
        dcnv.preprocess = lambda *args, **kwargs: None
        modules = {
            "suite2p": suite2p,
            "suite2p.detection": detection,
            "suite2p.detection.stats": stats,
            "suite2p.extraction": extraction,
            "suite2p.io": io_module,
            "suite2p.io.binary": binary,
            "suite2p.extraction.dcnv": dcnv,
        }
        with patch.dict(sys.modules, modules):
            api = load_suite2p_functions()
        self.assertEqual(api["roi_stats_path"], "suite2p.detection.stats.roi_stats")


if __name__ == "__main__":
    unittest.main()
