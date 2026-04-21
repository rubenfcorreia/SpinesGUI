#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


NORMAL_FILENAME = "ROIs_normal_mode_conversion.npy"
DEND_AXON_FILENAME = "ROIs_dendrite_axon_mode_conversion.npy"


def repo_root(user_id: str, animal_id: str, exp_id: str) -> Path:
    return Path(f"/home/{user_id}/data/Repository/{animal_id}/{exp_id}/suite2p")


def parse_exp_date(exp_id: str) -> str:
    return exp_id.split("_", 1)[0]


def find_same_day_candidates(user_id: str, animal_id: str, exp_id: str) -> list[Path]:
    animal_root = Path(f"/home/{user_id}/data/Repository/{animal_id}")
    if not animal_root.exists():
        return []
    date_prefix = parse_exp_date(exp_id)
    candidates = []
    for child in sorted(animal_root.iterdir()):
        if child.is_dir() and child.name.startswith(date_prefix + "_"):
            candidates.append(child / "suite2p")
    return candidates


def detect_conversion_file(suite2p_dir: Path) -> Optional[Path]:
    spines_dir = suite2p_dir / "SpinesGUI"
    if not spines_dir.exists():
        return None
    dend = spines_dir / DEND_AXON_FILENAME
    normal = spines_dir / NORMAL_FILENAME
    if dend.exists():
        return dend
    if normal.exists():
        return normal
    return None


def locate_conversion_file(user_id: str, animal_id: str, exp_id: str) -> Tuple[Optional[Path], Optional[str], bool]:
    primary = repo_root(user_id, animal_id, exp_id)
    conv = detect_conversion_file(primary)
    if conv is not None:
        return conv, exp_id, False

    for candidate in find_same_day_candidates(user_id, animal_id, exp_id):
        candidate_exp_id = candidate.parent.name
        if candidate_exp_id == exp_id:
            continue
        conv = detect_conversion_file(candidate)
        if conv is not None:
            return conv, candidate_exp_id, True
    return None, None, False


def detect_mode(path: Path) -> str:
    if path.name == DEND_AXON_FILENAME:
        return "dendrite_axon"
    if path.name == NORMAL_FILENAME:
        return "normal"
    raise ValueError(f"Unknown conversion filename: {path.name}")


def load_library(path: Path) -> Dict[Any, Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    if hasattr(data, "item"):
        data = data.item()
    if not isinstance(data, dict):
        raise TypeError("Loaded conversion file is not a dictionary")
    return data


def normalize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in value.items()}
    return value


def summarize(library: Dict[Any, Dict[str, Any]], mode: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "mode": mode,
        "total_rois": len(library),
        "general_roi_ids": sorted([int(k) if str(k).isdigit() else k for k in library.keys()], key=str),
        "roi_type_counts": {},
        "conversion_index_count": 0,
        "plane_counts": {},
    }
    for _, entry in library.items():
        entry = entry or {}
        roi_type_list = entry.get("roi-type")
        if isinstance(roi_type_list, (list, tuple)) and len(roi_type_list) >= 1:
            code = normalize_value(roi_type_list[0])
            out["roi_type_counts"][str(code)] = out["roi_type_counts"].get(str(code), 0) + 1
        if "conversion_index" in entry:
            out["conversion_index_count"] += 1
        plane = entry.get("plane")
        if plane is not None:
            plane = str(normalize_value(plane))
            out["plane_counts"][plane] = out["plane_counts"].get(plane, 0) + 1
    return out


def find_roi_by_general_id(library: Dict[Any, Dict[str, Any]], roi_id: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
    if roi_id in library:
        return roi_id, library[roi_id]
    try:
        int_id = int(roi_id)
    except ValueError:
        return None
    if int_id in library:
        return int_id, library[int_id]
    if str(int_id) in library:
        return str(int_id), library[str(int_id)]
    return None


def find_roi_by_conversion_index(library: Dict[Any, Dict[str, Any]], conversion_index: int) -> Optional[Tuple[Any, Dict[str, Any]]]:
    for key, entry in library.items():
        if normalize_value(entry.get("conversion_index")) == conversion_index:
            return key, entry
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a SpinesGUI conversion library")
    parser.add_argument("user_id")
    parser.add_argument("animal_id")
    parser.add_argument("exp_id")
    parser.add_argument("--general-roi-id")
    parser.add_argument("--conversion-index", type=int)
    args = parser.parse_args()

    conv_path, used_exp_id, used_fallback = locate_conversion_file(args.user_id, args.animal_id, args.exp_id)
    if conv_path is None:
        raise SystemExit(json.dumps({
            "found": False,
            "requested_exp_id": args.exp_id,
            "message": "No conversion library found for the requested experiment or same-day fallback"
        }, indent=2))

    mode = detect_mode(conv_path)
    library = load_library(conv_path)
    result: Dict[str, Any] = {
        "found": True,
        "path": str(conv_path),
        "mode": mode,
        "requested_exp_id": args.exp_id,
        "used_exp_id": used_exp_id,
        "used_same_day_fallback": used_fallback,
        "summary": summarize(library, mode),
    }

    if args.general_roi_id is not None:
        match = find_roi_by_general_id(library, args.general_roi_id)
        result["roi_lookup"] = None if match is None else {
            "general_roi_id": normalize_value(match[0]),
            "entry": normalize_value(match[1]),
        }

    if args.conversion_index is not None:
        match = find_roi_by_conversion_index(library, args.conversion_index)
        result["conversion_index_lookup"] = None if match is None else {
            "general_roi_id": normalize_value(match[0]),
            "entry": normalize_value(match[1]),
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
