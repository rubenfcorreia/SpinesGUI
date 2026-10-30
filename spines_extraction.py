# spines_extraction.py
from __future__ import annotations

import os
import sys
import io
import shutil
import tempfile
from typing import Callable, Optional

import numpy as np

from suite2p_compat import (
    load_suite2p_functions,
    normalize_suite2p_ops,
    extraction_wrapper_compat,
    roi_stats_compat,
    suite2p_config_candidates,
)
from suite2p_numpy_compat import load_suite2p_dict, load_suite2p_npy


def _load_required_dict(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return load_suite2p_dict(path)


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _delete_if_exists(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[DEBUG] Failed to delete {path}: {e}", flush=True)


def _copy_if_incomplete(source_path: str, destination_path: str) -> bool:
    """Copy a binary when its destination is missing or has a different size."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Missing source binary: {source_path}")

    source_size = os.path.getsize(source_path)
    if os.path.exists(destination_path) and os.path.getsize(destination_path) == source_size:
        print(f"[DEBUG] Binary copy complete; skipping {destination_path}", flush=True)
        return False

    if not os.path.exists(destination_path):
        reason = "missing destination"
    else:
        reason = (
            f"size mismatch (source={source_size}, "
            f"destination={os.path.getsize(destination_path)})"
        )
    print(f"[DEBUG] Copying {source_path} → {destination_path}: {reason}", flush=True)

    destination_dir = os.path.dirname(destination_path) or "."
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination_path)}.tmp-",
        dir=destination_dir,
    )
    os.close(fd)
    try:
        with open(source_path, "rb") as source, open(temporary_path, "wb") as temporary:
            shutil.copyfileobj(source, temporary, length=16 * 1024)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination_path)
    except Exception:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return True


def _has_extraction_outputs(plane_folder: str) -> bool:
    """Return whether fluorescence extraction completed for this plane."""
    required = ("stat.npy", "F.npy", "Fneu.npy", "F_chan2.npy", "Fneu_chan2.npy")
    return all(os.path.exists(os.path.join(plane_folder, name)) for name in required)


class ExtractionCanceled(Exception):
    """Raised when the queue requests a safe stop between extraction stages."""


def run_extraction(root_folder: str, mode: str, force: bool, log_path: Optional[str] = None,
                   cancel_check: Optional[Callable[[], bool]] = None) -> None:
    """
    Headless extraction entry point (runs in worker).
    - No Qt imports
    - Uses print() only (worker tee captures logs)
    - Raises exceptions on failure
    """

    print(f"[extract] root_folder={root_folder}", flush=True)
    print(f"[extract] mode={mode} force={force}", flush=True)

    def check_canceled() -> None:
        if cancel_check and cancel_check():
            print("[extract] cancellation requested; stopping at safe checkpoint", flush=True)
            raise ExtractionCanceled()

    check_canceled()
    spines_gui_folder = os.path.join(root_folder, "SpinesGUI")
    _safe_makedirs(spines_gui_folder)

    # ---- Load data saved by GUI ----
    roi_data_path = os.path.join(spines_gui_folder, "roi_data.npy")
    plane_data_path = os.path.join(spines_gui_folder, "plane_data.npy")
    roi_data = _load_required_dict(roi_data_path)
    plane_data = _load_required_dict(plane_data_path)

    # ---- Imports from suite2p (v1 first, legacy paths supported) ----
    suite2p_api = load_suite2p_functions()
    extraction_wrapper = suite2p_api["extraction_wrapper"]
    BinaryFile = suite2p_api["BinaryFile"]
    oasis = suite2p_api["oasis"]
    preprocess = suite2p_api["preprocess"]

    success_file = os.path.join(spines_gui_folder, "extraction_successfull.txt")

    # If already extracted and not forcing, bail out
    if os.path.exists(success_file) and not force:
        raise RuntimeError(
            f"Extraction already done for {root_folder}. Pass force=True to re-run."
        )

    # ---- If forcing, delete prior outputs (but keep data.bin, data_chan2.bin, logs) ----
    check_canceled()
    if force:
        print("[DEBUG] force=True → cleaning previous extraction outputs", flush=True)
        for plane in plane_data.keys():
            plane_folder = os.path.join(spines_gui_folder, f"plane{plane}")
            for fname in [
                "stat0.npy", "stat1.npy", "stat.npy", "F.npy", "Fneu.npy",
                "F_chan2.npy", "Fneu_chan2.npy", "spks.npy", "iscell.npy",
                "stat0_dendrite_axon_mode.npy", "stat1_dendrite_axon_mode.npy",
            ]:
                _delete_if_exists(os.path.join(plane_folder, fname))
        _delete_if_exists(success_file)

    # ---- Main per-plane extraction ----
    for plane in plane_data.keys():
        check_canceled()
        print(f"[DEBUG] Processing extraction for plane {plane}", flush=True)

        plane_folder = os.path.join(spines_gui_folder, f"plane{plane}")
        _safe_makedirs(plane_folder)

        # Ensure binaries are present in SpinesGUI/planeX (copy from original plane folder if needed)
        src_plane_folder = plane_data[plane]["folder"]

        # data.bin
        data_bin_src = os.path.join(src_plane_folder, "data.bin")
        data_bin_dest = os.path.join(plane_folder, "data.bin")
        _copy_if_incomplete(data_bin_src, data_bin_dest)

        # data_chan2.bin (optional)
        data_chan2_src = os.path.join(src_plane_folder, "data_chan2.bin")
        data_chan2_dest = os.path.join(plane_folder, "data_chan2.bin")
        if os.path.exists(data_chan2_src):
            _copy_if_incomplete(data_chan2_src, data_chan2_dest)
        else:
            data_chan2_dest = None  # keep your logic

        # Suite2p v1 uses settings/config files; legacy runs use ops.npy.
        config_candidates = suite2p_config_candidates(src_plane_folder)
        if not config_candidates:
            raise FileNotFoundError(
                f"No Suite2p config found in {src_plane_folder}; expected "
                "settings.npy, ops.npy, or db.npy"
            )
        ops_dest = os.path.join(plane_folder, "ops.npy")
        configs = [load_suite2p_dict(path) for path in config_candidates]
        ops = {}
        for config in configs:
            ops.update(config)
        ops = normalize_suite2p_ops(ops)
        ops["ops_path"] = ops_dest
        if "reg_file" in ops:
            ops["reg_file"] = os.path.join(plane_folder, os.path.basename(ops["reg_file"]))
            print(f"[DEBUG] Patched ops['reg_file'] → {ops['reg_file']}", flush=True)
        if "reg_file_chan2" in ops:
            ops["reg_file_chan2"] = os.path.join(plane_folder, os.path.basename(ops["reg_file_chan2"]))
            print(f"[DEBUG] Patched ops['reg_file_chan2'] → {ops['reg_file_chan2']}", flush=True)
        np.save(ops_dest, ops)

        # Extract parameters
        Ly = ops.get("Ly")
        Lx = ops.get("Lx")
        aspect = ops.get("aspect", 1.0)
        diameter = ops.get("diameter", 10)
        max_overlap = ops.get("max_overlap", 1.0)
        do_crop = ops.get("soma_crop", 1)

        print(
            f"[DEBUG] Plane {plane} params: Ly={Ly}, Lx={Lx}, aspect={aspect}, "
            f"diameter={diameter}, max_overlap={max_overlap}, do_crop={do_crop}",
            flush=True
        )

        if _has_extraction_outputs(plane_folder) and not force:
            print(f"[DEBUG] Plane {plane}: extraction outputs exist; skipping ROI extraction", flush=True)
            stat_out = load_suite2p_npy(os.path.join(plane_folder, "stat.npy"))
            F = load_suite2p_npy(os.path.join(plane_folder, "F.npy"))
            Fneu = load_suite2p_npy(os.path.join(plane_folder, "Fneu.npy"))
        else:
            # ---- Build stat0 from roi_data ----
            stat0 = {}
            roi_list = [(k, roi) for k, roi in roi_data.items() if roi.get("plane") == plane]
            roi_list_sorted = sorted(roi_list, key=lambda x: x[0])

            from matplotlib.path import Path  # local import ok

            for idx, (roi_key, roi) in enumerate(roi_list_sorted):
                vertices = np.array(roi.get("ROI coordinates", []))
                if vertices.size == 0:
                    print(f"[DEBUG] ROI {roi_key} on plane {plane} has no vertices.", flush=True)
                    continue

                x_min = int(np.floor(np.min(vertices[:, 0])))
                x_max = int(np.ceil(np.max(vertices[:, 0])))
                y_min = int(np.floor(np.min(vertices[:, 1])))
                y_max = int(np.ceil(np.max(vertices[:, 1])))

                xx, yy = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
                points = np.vstack((xx.flatten(), yy.flatten())).T
                inside = Path(vertices).contains_points(points).reshape(yy.shape)
                ypix = np.where(inside)[0] + y_min
                xpix = np.where(inside)[1] + x_min
                lam = np.ones(ypix.shape)
                stat0[idx] = {"ypix": np.array(ypix), "xpix": np.array(xpix), "lam": np.array(lam)}
                print(f"[DEBUG] Plane {plane}, ROI index {idx}: mask {len(ypix)} px", flush=True)

            np.save(os.path.join(plane_folder, "stat0.npy"), stat0)
            stat0_arr = np.array(list(stat0.values()), dtype=object)
            diameter_values = np.asarray(diameter if diameter is not None else [], dtype=float).reshape(-1)
            if diameter_values.size == 0 or np.all(diameter_values <= 0):
                print(f"[DEBUG] Plane {plane}: ops diameter={diameter} → using default 10", flush=True)
                diameter = 10

            stat1 = roi_stats_compat(
                stat0_arr, Ly, Lx, aspect=aspect, diameter=diameter,
                max_overlap=max_overlap, do_crop=do_crop,
                roi_stats=suite2p_api["roi_stats"],
            )
            stat1_filename = "stat1.npy" if mode == "normal" else "stat1_dendrite_axon_mode.npy"
            np.save(os.path.join(plane_folder, stat1_filename), stat1)

            f_reg_data = BinaryFile(
                Ly, Lx, data_bin_dest, n_frames=ops.get("nframes"),
                dtype=ops.get("datatype", "int16"),
            )
            if data_chan2_dest is not None:
                f_reg_chan2_data = BinaryFile(
                    Ly, Lx, data_chan2_dest, n_frames=ops.get("nframes"),
                    dtype=ops.get("datatype", "int16"),
                )
            else:
                f_reg_chan2_data = None

            lam_percentile_override = None
            if mode == "dendrites_axons":
                lam_percentile_override = 0.0
                print(
                    "[DEBUG] dendrites_axons mode: disabling lam_percentile "
                    "filter for manual ROI masks",
                    flush=True,
                )

            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()
            try:
                outputs = extraction_wrapper_compat(
                    extraction_wrapper, stat1, f_reg_data, f_reg_chan2_data, ops,
                    lam_percentile_override=lam_percentile_override,
                )
            finally:
                sys.stdout = old_stdout

            extraction_printed = mystdout.getvalue()
            if extraction_printed.strip():
                print("[DEBUG] extraction_wrapper printed:", flush=True)
                print(extraction_printed, flush=True)

            check_canceled()
            stat_out, F, Fneu, F_chan2, Fneu_chan2 = outputs
            np.save(os.path.join(plane_folder, "stat.npy"), stat_out)
            np.save(os.path.join(plane_folder, "F.npy"), F)
            np.save(os.path.join(plane_folder, "Fneu.npy"), Fneu)
            np.save(os.path.join(plane_folder, "F_chan2.npy"), F_chan2)
            np.save(os.path.join(plane_folder, "Fneu_chan2.npy"), Fneu_chan2)

        check_canceled()
        # ---- Spike deconvolution ----
        print(f"[DEBUG] Running spike deconvolution for plane {plane}", flush=True)
        dF = F.copy() - ops["neucoeff"] * Fneu
        dF = preprocess(
            F=dF,
            baseline=ops["baseline"],
            win_baseline=ops["win_baseline"],
            sig_baseline=ops["sig_baseline"],
            fs=ops["fs"],
            prctile_baseline=ops["prctile_baseline"],
        )
        spks = oasis(F=dF, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"])
        check_canceled()
        np.save(os.path.join(plane_folder, "spks.npy"), spks)

        # ---- iscell ----
        roi_ids = [roi_id for roi_id, info in roi_data.items() if info.get("plane") == plane]
        iscell_arr = np.ones((len(roi_ids), 2), dtype=int)
        np.save(os.path.join(plane_folder, "iscell.npy"), iscell_arr)

    check_canceled()
    # ---- Success file ----
    with open(success_file, "w", encoding="utf-8") as sf:
        sf.write("Extraction finished successfully.\n")
    print(f"[DEBUG] Created extraction success file at {success_file}", flush=True)

    # ---- Conversion dict (save only; no GUI display) ----
    conversion_dict = {}
    plane_groups = {}
    for key, roi in roi_data.items():
        if mode == "dendrites_axons" and len(roi.get("roi-type", [])) < 5:
            roi["roi-type"] = roi.get("roi-type", []) + [0]
        p = roi.get("plane")
        plane_groups.setdefault(p, []).append((key, roi))

    for plane, items in plane_groups.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        for idx, (roi_key, roi) in enumerate(items_sorted):
            roi["conversion"] = [plane, idx]
            conversion_dict[roi_key] = roi

    sorted_conversion = sorted(
        conversion_dict.items(),
        key=lambda x: (x[1]["conversion"][0], x[1]["conversion"][1])
    )
    for new_index, (roi_key, roi) in enumerate(sorted_conversion):
        roi["conversion index"] = new_index

    conv_filename = "ROIs_conversion.npy" if mode == "normal" else "ROIs_dendrite_axon_mode_conversion.npy"
    rois_conv_file = os.path.join(spines_gui_folder, conv_filename)
    np.save(rois_conv_file, conversion_dict)
    print(f"[DEBUG] Saved conversion dictionary to {rois_conv_file}", flush=True)

    print("[extract] finished successfully", flush=True)
