# spines_extraction.py
from __future__ import annotations

import os
import sys
import io
import shutil
from typing import Optional

import numpy as np


def _load_required_dict(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    obj = np.load(path, allow_pickle=True)
    # np.save(dict) loads as 0-d ndarray; .item() gives dict
    return obj.item()


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _delete_if_exists(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[DEBUG] Failed to delete {path}: {e}", flush=True)


def run_extraction(root_folder: str, mode: str, force: bool, log_path: Optional[str] = None) -> None:
    """
    Headless extraction entry point (runs in worker).
    - No Qt imports
    - Uses print() only (worker tee captures logs)
    - Raises exceptions on failure
    """

    print(f"[extract] root_folder={root_folder}", flush=True)
    print(f"[extract] mode={mode} force={force}", flush=True)

    spines_gui_folder = os.path.join(root_folder, "SpinesGUI")
    _safe_makedirs(spines_gui_folder)

    # ---- Load data saved by GUI ----
    roi_data_path = os.path.join(spines_gui_folder, "roi_data.npy")
    plane_data_path = os.path.join(spines_gui_folder, "plane_data.npy")
    roi_data = _load_required_dict(roi_data_path)
    plane_data = _load_required_dict(plane_data_path)

    # ---- Imports from suite2p ----
    from suite2p.detection import roi_stats  # noqa
    from suite2p.extraction import extraction_wrapper  # noqa
    from suite2p.io.binary import BinaryFile  # noqa
    from suite2p.extraction.dcnv import oasis, preprocess  # noqa

    success_file = os.path.join(spines_gui_folder, "extraction_successfull.txt")

    # If already extracted and not forcing, bail out
    if os.path.exists(success_file) and not force:
        raise RuntimeError(
            f"Extraction already done for {root_folder}. Pass force=True to re-run."
        )

    # ---- If forcing, delete prior outputs (but keep data.bin, data_chan2.bin, logs) ----
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
        print(f"[DEBUG] Processing extraction for plane {plane}", flush=True)

        plane_folder = os.path.join(spines_gui_folder, f"plane{plane}")
        _safe_makedirs(plane_folder)

        # Ensure binaries are present in SpinesGUI/planeX (copy from original plane folder if needed)
        src_plane_folder = plane_data[plane]["folder"]

        # data.bin
        data_bin_src = os.path.join(src_plane_folder, "data.bin")
        data_bin_dest = os.path.join(plane_folder, "data.bin")
        if not os.path.exists(data_bin_dest):
            print(f"[DEBUG] Copying data.bin → {data_bin_dest}", flush=True)
            with open(data_bin_src, "rb") as fsrc, open(data_bin_dest, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, length=16 * 1024)

        # data_chan2.bin (optional)
        data_chan2_src = os.path.join(src_plane_folder, "data_chan2.bin")
        data_chan2_dest = os.path.join(plane_folder, "data_chan2.bin")
        if os.path.exists(data_chan2_src) and not os.path.exists(data_chan2_dest):
            print(f"[DEBUG] Copying data_chan2.bin → {data_chan2_dest}", flush=True)
            with open(data_chan2_src, "rb") as fsrc2, open(data_chan2_dest, "wb") as fdst2:
                shutil.copyfileobj(fsrc2, fdst2, length=16 * 1024)
        if not os.path.exists(data_chan2_src):
            data_chan2_dest = None  # keep your logic

        # ops.npy (copy + patch paths)
        ops_src = os.path.join(src_plane_folder, "ops.npy")
        ops_dest = os.path.join(plane_folder, "ops.npy")
        shutil.copy(ops_src, ops_dest)

        ops = np.load(ops_dest, allow_pickle=True).item()
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
        if isinstance(aspect, (list, tuple, np.ndarray)):
            aspect = aspect[0]
        diameter = ops.get("diameter", 10)
        if isinstance(diameter, (list, tuple, np.ndarray)):
            diameter = diameter[0]
        max_overlap = ops.get("max_overlap", 1.0)
        if isinstance(max_overlap, (list, tuple, np.ndarray)):
            max_overlap = max_overlap[0]
        do_crop = ops.get("soma_crop", 1)
        if isinstance(do_crop, (list, tuple, np.ndarray)):
            do_crop = do_crop[0]

        print(
            f"[DEBUG] Plane {plane} params: Ly={Ly}, Lx={Lx}, aspect={aspect}, "
            f"diameter={diameter}, max_overlap={max_overlap}, do_crop={do_crop}",
            flush=True
        )

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

        stat0_file = os.path.join(plane_folder, "stat0.npy")
        np.save(stat0_file, stat0)

        # roi_stats expects list
        stat0_list = list(stat0.values())

        stat1 = roi_stats(
            stat0_list, Ly, Lx,
            aspect=aspect, diameter=diameter, max_overlap=max_overlap, do_crop=do_crop
        )
        stat1_filename = "stat1.npy" if mode == "normal" else "stat1_dendrite_axon_mode.npy"
        stat1_file = os.path.join(plane_folder, stat1_filename)
        np.save(stat1_file, stat1)

        # ---- Open binaries and run extraction_wrapper ----
        f_reg_data = BinaryFile(Ly, Lx, data_bin_dest, n_frames=ops.get("nframes"), dtype=ops.get("datatype", "int16"))

        if data_chan2_dest is not None:
            f_reg_chan2_data = BinaryFile(Ly, Lx, data_chan2_dest, n_frames=ops.get("nframes"), dtype=ops.get("datatype", "int16"))
        else:
            f_reg_chan2_data = None

        # If you want to capture extraction_wrapper prints, keep this.
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        try:
            outputs = extraction_wrapper(
                stat1, f_reg_data, f_reg_chan2_data,
                cell_masks=None, neuropil_masks=None, ops=ops
            )
        finally:
            sys.stdout = old_stdout

        extraction_printed = mystdout.getvalue()
        if extraction_printed.strip():
            print("[DEBUG] extraction_wrapper printed:", flush=True)
            print(extraction_printed, flush=True)

        stat_out, F, Fneu, F_chan2, Fneu_chan2 = outputs
        np.save(os.path.join(plane_folder, "stat.npy"), stat_out)
        np.save(os.path.join(plane_folder, "F.npy"), F)
        np.save(os.path.join(plane_folder, "Fneu.npy"), Fneu)
        np.save(os.path.join(plane_folder, "F_chan2.npy"), F_chan2)
        np.save(os.path.join(plane_folder, "Fneu_chan2.npy"), Fneu_chan2)

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
        np.save(os.path.join(plane_folder, "spks.npy"), spks)

        # ---- iscell ----
        roi_ids = [roi_id for roi_id, info in roi_data.items() if info.get("plane") == plane]
        iscell_arr = np.ones((len(roi_ids), 2), dtype=int)
        np.save(os.path.join(plane_folder, "iscell.npy"), iscell_arr)

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
