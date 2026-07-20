"""Suite2p NumPy pickle compatibility helpers.

Suite2p v1 / NumPy 2 can write object pickles that reference private module
paths such as ``numpy._core.multiarray`` and ``numpy._core.numeric``. The
legacy SpinesGUI runtime pins NumPy 1.24, which cannot import those paths
directly. We only install compatibility aliases when a trusted local Suite2p
or SpinesGUI object-array ``.npy`` load fails for that reason, then retry the
load.

The aliases are idempotent and only fill missing ``sys.modules`` entries.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Union

import numpy as np

_NUMPY_CORE_ALIAS_TARGETS = (
    ("numpy._core", "numpy.core"),
    ("numpy._core.multiarray", "numpy.core.multiarray"),
    ("numpy._core.numeric", "numpy.core.numeric"),
)


def _is_numpy_core_pickle_error(exc: BaseException) -> bool:
    """Return True when NumPy 2 private pickle paths are the missing import."""
    module_name = getattr(exc, "name", None)
    if isinstance(module_name, str) and module_name.startswith("numpy._core"):
        return True
    return "numpy._core" in str(exc)


def _register_alias(alias: str, target: str) -> bool:
    """Install a sys.modules alias without overwriting an existing module."""
    existing = sys.modules.get(alias)
    if existing is not None:
        return False

    module = importlib.import_module(target)
    sys.modules.setdefault(alias, module)
    return True


def ensure_suite2p_numpy_pickle_compat() -> None:
    """Install NumPy 2 pickle aliases needed for trusted Suite2p loads."""
    for alias, target in _NUMPY_CORE_ALIAS_TARGETS:
        _register_alias(alias, target)


def load_suite2p_npy(
    path: Union[os.PathLike[str], str],
    *,
    allow_pickle: bool = True,
    **kwargs: Any,
):
    """Load a trusted Suite2p ``.npy`` file with a narrow compatibility retry."""
    file_path = os.fspath(path)
    try:
        return np.load(file_path, allow_pickle=allow_pickle, **kwargs)
    except (ModuleNotFoundError, ImportError) as exc:
        if not _is_numpy_core_pickle_error(exc):
            raise

    ensure_suite2p_numpy_pickle_compat()

    try:
        return np.load(file_path, allow_pickle=allow_pickle, **kwargs)
    except (ModuleNotFoundError, ImportError) as exc:
        if _is_numpy_core_pickle_error(exc):
            raise RuntimeError(
                f"{file_path} still references NumPy 2 private modules after "
                "installing Suite2p compatibility aliases. The file may be "
                "corrupt or unsupported."
            ) from exc
        raise


def load_suite2p_dict(
    path: Union[os.PathLike[str], str],
    *,
    allow_pickle: bool = True,
    **kwargs: Any,
) -> dict:
    """Load a pickled Suite2p dictionary from a 0-d object array."""
    loaded = load_suite2p_npy(path, allow_pickle=allow_pickle, **kwargs)
    if not isinstance(loaded, np.ndarray):
        raise TypeError(f"Expected a NumPy array from {path!r}, got {type(loaded).__name__}.")
    if loaded.ndim != 0:
        raise TypeError(
            f"Expected a 0-d object array from {path!r}, got shape {loaded.shape}."
        )

    try:
        payload = loaded.item()
    except ValueError as exc:
        raise TypeError(f"{path!r} did not contain a single pickled object.") from exc

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a dictionary in {path!r}, got {type(payload).__name__}."
        )
    return payload
