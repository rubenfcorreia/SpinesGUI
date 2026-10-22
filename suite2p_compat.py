"""Compatibility helpers for legacy Suite2p and Suite2p v1.

This module intentionally has no Qt dependency.  Suite2p v0 exposed
``roi_stats`` from ``suite2p.detection`` and accepted the legacy keyword
``do_crop``.  Suite2p v1 exposes it from ``suite2p.detection.stats`` and uses
``do_soma_crop`` plus a two-element diameter.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from typing import Any, Callable, Mapping

import numpy as np


def _import_first(paths: tuple[tuple[str, str], ...]) -> tuple[Any, str]:
    errors = []
    for module_name, attribute in paths:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attribute), f"{module_name}.{attribute}"
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}.{attribute}: {exc}")
    raise ImportError("Could not import Suite2p compatibility symbol:\n" + "\n".join(errors))


def load_suite2p_functions() -> dict[str, Any]:
    """Load Suite2p functions lazily and report the resolved API paths."""
    roi_stats, roi_stats_path = _import_first(
        (("suite2p.detection", "roi_stats"), ("suite2p.detection.stats", "roi_stats"))
    )
    extraction_wrapper, extraction_path = _import_first(
        (("suite2p.extraction", "extraction_wrapper"),)
    )
    binary_file, binary_path = _import_first((("suite2p.io.binary", "BinaryFile"),))
    oasis, oasis_path = _import_first((("suite2p.extraction.dcnv", "oasis"),))
    preprocess, preprocess_path = _import_first((("suite2p.extraction.dcnv", "preprocess"),))

    try:
        suite2p = importlib.import_module("suite2p")
        version = getattr(suite2p, "__version__", "unknown")
        suite2p_path = getattr(suite2p, "__file__", "unknown")
    except ImportError:
        version, suite2p_path = "unknown", "unknown"

    print(
        f"[suite2p] version={version} path={suite2p_path} "
        f"python={sys.executable} roi_stats={roi_stats_path}",
        flush=True,
    )
    return {
        "roi_stats": roi_stats,
        "extraction_wrapper": extraction_wrapper,
        "BinaryFile": binary_file,
        "oasis": oasis,
        "preprocess": preprocess,
        "roi_stats_path": roi_stats_path,
        "extraction_path": extraction_path,
        "binary_path": binary_path,
        "oasis_path": oasis_path,
        "preprocess_path": preprocess_path,
    }


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.reshape(-1)[0]
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _nested_value(config: Mapping[str, Any], *keys: str) -> Any:
    """Find a setting in flat, common nested, or recursively nested config."""
    for key in keys:
        if key in config:
            return config[key]
    for value in config.values():
        if isinstance(value, Mapping):
            found = _nested_value(value, *keys)
            if found is not None:
                return found
    return None


def normalize_suite2p_ops(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return flat extraction settings for both legacy and v1 config shapes."""
    ops = dict(config)
    for key in ("Ly", "Lx", "nframes", "datatype", "neucoeff", "baseline",
                "win_baseline", "sig_baseline", "fs", "batch_size", "tau",
                "reg_file", "reg_file_chan2"):
        if key not in ops:
            value = _nested_value(config, key)
            if value is not None:
                ops[key] = value

    if ops.get("neucoeff") is None:
        value = _nested_value(config, "neuropil_coefficient")
        if value is not None:
            ops["neucoeff"] = value

    diameter = _nested_value(config, "diameter")
    aspect = _nested_value(config, "aspect")
    max_overlap = _nested_value(config, "max_overlap")
    do_crop = _nested_value(config, "soma_crop", "do_soma_crop", "do_crop")
    if diameter is not None:
        ops["diameter"] = diameter
    if aspect is not None:
        ops["aspect"] = aspect
    if max_overlap is not None:
        ops["max_overlap"] = max_overlap
    if do_crop is not None:
        ops["soma_crop"] = do_crop

    ops["Ly"] = _scalar(ops.get("Ly"))
    ops["Lx"] = _scalar(ops.get("Lx"))
    ops["aspect"] = float(_scalar(ops.get("aspect"), 1.0))
    diameter_value = ops.get("diameter")
    if isinstance(diameter_value, np.ndarray):
        diameter_value = diameter_value.tolist()
    if not isinstance(diameter_value, (list, tuple)):
        diameter_value = _scalar(diameter_value, 10)
    ops["diameter"] = diameter_value
    ops["max_overlap"] = float(_scalar(ops.get("max_overlap"), 1.0))
    ops["soma_crop"] = bool(_scalar(ops.get("soma_crop"), True))
    if ops.get("prctile_baseline") is None:
        ops["prctile_baseline"] = 8
    return ops


def _diameter_pair(diameter: Any, aspect: Any) -> list[float]:
    if diameter is None:
        diameter = 10
    if isinstance(diameter, np.ndarray):
        diameter = diameter.tolist()
    if isinstance(diameter, (list, tuple)):
        values = [float(v) for v in diameter]
        if len(values) == 1:
            values *= 2
        return values[:2]
    dx = float(diameter)
    return [float(aspect if aspect is not None else 1.0) * dx, dx]


def _required_roi_fields(stat: Any) -> bool:
    required = {"ypix", "xpix", "lam"}
    try:
        return all(required.issubset(set(item)) for item in stat)
    except TypeError:
        return False


class _FallbackROI:
    def __init__(self, item: Mapping[str, Any], do_crop: bool):
        self.ypix = np.asarray(item["ypix"])
        self.xpix = np.asarray(item["xpix"])
        self.lam = np.asarray(item["lam"])
        self.med = item.get("med", [np.median(self.ypix), np.median(self.xpix)])
        self.do_crop = do_crop
        self.soma_crop = np.ones(self.ypix.shape, dtype=bool)


def _fallback_roi_stats(stat: Any, Ly: int, Lx: int, *, aspect: Any,
                        diameter: Any, max_overlap: Any, do_crop: bool) -> Any:
    """Minimal safe fallback for manually drawn masks unsupported by Suite2p."""
    items = list(stat)
    rois = [_FallbackROI(item, do_crop) for item in items]
    overlap = np.zeros((int(Ly), int(Lx)), dtype=float)
    for roi in rois:
        overlap[roi.ypix, roi.xpix] += 1
    pair = _diameter_pair(diameter, aspect)
    for roi, item in zip(rois, items):
        item.setdefault("med", roi.med)
        item["mrs"] = 1.0
        item["mrs0"] = 1.0
        item["compact"] = 1.0
        item["solidity"] = 1.0
        item["npix"] = len(roi.ypix)
        item["npix_soma"] = len(roi.ypix)
        item["soma_crop"] = roi.soma_crop
        item["overlap"] = overlap[roi.ypix, roi.xpix] > 1
        item["radius"] = float(np.mean(pair) / 2.0)
        item["aspect_ratio"] = float(pair[0] / pair[1]) if pair[1] else 1.0
        item["npix_norm_no_crop"] = 1.0
        item["npix_norm"] = 1.0
        item.setdefault("footprint", 0)
    if max_overlap is not None and float(max_overlap) < 1.0:
        items = [item for item in items if not np.mean(item["overlap"]) > float(max_overlap)]
    return np.asarray(items, dtype=object)


def roi_stats_compat(stat, Ly: int, Lx: int, *, aspect=None, diameter=None,
                     max_overlap=None, do_crop=True, roi_stats: Callable | None = None):
    """Call either Suite2p ROI-statistics API with canonical arguments."""
    if not _required_roi_fields(stat):
        raise ValueError("ROI statistics require ypix, xpix, and lam fields")
    if roi_stats is None:
        roi_stats = load_suite2p_functions()["roi_stats"]

    params = inspect.signature(roi_stats).parameters
    kwargs: dict[str, Any] = {}
    if "diameter" in params:
        kwargs["diameter"] = (
            _diameter_pair(diameter, aspect)
            if "do_soma_crop" in params else diameter
        )
    if "aspect" in params:
        kwargs["aspect"] = aspect
    if "max_overlap" in params:
        kwargs["max_overlap"] = max_overlap
    if "do_crop" in params:
        kwargs["do_crop"] = do_crop
    elif "do_soma_crop" in params:
        kwargs["do_soma_crop"] = do_crop

    try:
        result = roi_stats(stat, Ly, Lx, **kwargs)
        if not _required_roi_fields(result):
            raise ValueError("Suite2p roi_stats returned invalid ROI records")
        return result
    except (TypeError, ValueError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
        print(f"[suite2p] native roi_stats fallback for manual masks: {exc}", flush=True)
        return _fallback_roi_stats(
            stat, Ly, Lx, aspect=aspect, diameter=diameter,
            max_overlap=max_overlap, do_crop=do_crop,
        )


def extraction_wrapper_compat(extraction_wrapper: Callable, stat: Any, f_reg: Any,
                               f_reg_chan2: Any, ops: Mapping[str, Any]) -> tuple:
    """Call legacy ``ops=`` or v1 ``settings=`` extraction APIs."""
    params = inspect.signature(extraction_wrapper).parameters
    kwargs: dict[str, Any] = {"cell_masks": None, "neuropil_masks": None}
    if "ops" in params:
        kwargs["ops"] = ops
    elif "settings" in params:
        settings = dict(ops.get("extraction", {})) if isinstance(ops.get("extraction"), Mapping) else {}
        settings.update({
            "batch_size": ops.get("batch_size", settings.get("batch_size", 500)),
            "neuropil_coefficient": ops.get("neucoeff", settings.get("neuropil_coefficient", 0.7)),
            "neuropil_extract": settings.get("neuropil_extract", True),
            "inner_neuropil_radius": settings.get("inner_neuropil_radius", 2),
            "min_neuropil_pixels": settings.get("min_neuropil_pixels", 350),
            "lam_percentile": settings.get("lam_percentile", 50.0),
            "allow_overlap": settings.get("allow_overlap", True),
            "circular_neuropil": settings.get("circular_neuropil", False),
        })
        kwargs["settings"] = settings
    else:
        raise TypeError(
            "Unsupported Suite2p extraction_wrapper signature; expected ops or settings"
        )

    outputs = extraction_wrapper(stat, f_reg, f_reg_chan2, **kwargs)
    if not isinstance(outputs, tuple):
        outputs = tuple(outputs)
    if len(outputs) == 4:
        return (stat, *outputs)
    if len(outputs) == 5:
        return outputs
    raise ValueError(
        f"Unsupported extraction_wrapper return value: expected 4 or 5 items, got {len(outputs)}"
    )


def suite2p_config_candidates(plane_folder: str) -> list[str]:
    return [
        os.path.join(plane_folder, name)
        for name in ("ops.npy", "settings.npy", "db.npy")
        if os.path.exists(os.path.join(plane_folder, name))
    ]
