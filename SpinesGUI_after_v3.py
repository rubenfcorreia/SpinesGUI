#!/usr/bin/env python3
"""
SpinesGUI_after_v3.py

Post-SpinesGUI "cleanup + split" helper.

What it does:
  1) Inside <exp_dir_processed>/suite2p/:
     - creates suite2p_original_files/
     - moves everything EXCEPT SpinesGUI/ and suite2p_original_files/ into suite2p_original_files/
     - copies each plane folder from suite2p/SpinesGUI/<planeX>/ back into suite2p/planeX/

  2) Patches every ops.npy under suite2p/**/ops.npy:
     - ops["ops_path"] set to the file's own path
     - ops["reg_file"] / ops["reg_file_chan2"] rebased into the same folder (basename preserved)

  3) If --combined is passed, runs the split of suite2p_combined into per-expID suite2p folders,
     with SpinesGUI artifact propagation (conversion libs, extraction logs, mode files).

Usage:
  python SpinesGUI_after_v3.py --userID rubencorreia --expID 2025-11-11_01_ESRC022 --combined

Notes:
  - This script assumes your split function lives in split_combined_s2p_modified_with_spinesgui.py
    and exposes split_combined_suite2p_v3().
"""

import os
import glob
import shutil
import numpy as np

import organise_paths
from split_combined_s2p_modified_with_spinesgui import split_combined_suite2p_v3


def move_original_suite2p_files(userID: str, expID: str) -> None:
    """
    1. Creates suite2p_original_files/ inside suite2p/
    2. Moves everything except SpinesGUI and suite2p_original_files into it
    3. Copies each plane-subfolder from SpinesGUI/ back into suite2p/
    """
    _, _, _, exp_dir_processed, _ = organise_paths.find_paths(userID, expID)
    suite2p_folder = os.path.join(exp_dir_processed, "suite2p")
    new_folder = os.path.join(suite2p_folder, "suite2p_original_files")
    os.makedirs(new_folder, exist_ok=True)
    print(f"[SpinesGUI_after] Created {new_folder!r}")

    if not os.path.isdir(suite2p_folder):
        raise FileNotFoundError(f"[SpinesGUI_after] suite2p folder not found: {suite2p_folder}")

    spines_gui = os.path.join(suite2p_folder, "SpinesGUI")
    if not os.path.isdir(spines_gui):
        raise FileNotFoundError(f"[SpinesGUI_after] SpinesGUI folder not found: {spines_gui}")

    # Move all top-level items except SpinesGUI and our new folder
    for item in os.listdir(suite2p_folder):
        if item in ("SpinesGUI", "suite2p_original_files"):
            continue
        src = os.path.join(suite2p_folder, item)
        print(f"[SpinesGUI_after] Moving {src!r} → {new_folder!r}")
        shutil.move(src, new_folder)

    # Copy each plane folder out of SpinesGUI back into suite2p/
    for plane in os.listdir(spines_gui):
        src_plane = os.path.join(spines_gui, plane)
        if not os.path.isdir(src_plane):
            continue
        dst_plane = os.path.join(suite2p_folder, plane)
        os.makedirs(dst_plane, exist_ok=True)
        print(f"[SpinesGUI_after] Copying contents of {src_plane!r} → {dst_plane!r}")
        for fn in os.listdir(src_plane):
            shutil.copy2(os.path.join(src_plane, fn), dst_plane)

    print("[SpinesGUI_after] All plane folders have been moved/copied into suite2p/.")


def patch_all_ops_paths(userID: str, expID: str) -> None:
    """
    Recursively find every ops.npy under suite2p/ and:
      - reset ops['ops_path'] to its own file path
      - rebase reg_file and reg_file_chan2 if they exist
      - save back out
    """
    _, _, _, exp_dir_processed, _ = organise_paths.find_paths(userID, expID)
    suite2p_folder = os.path.join(exp_dir_processed, "suite2p")

    pattern = os.path.join(suite2p_folder, "**", "ops.npy")
    ops_files = glob.glob(pattern, recursive=True)
    print(f"[SpinesGUI_after] Found {len(ops_files)} ops.npy files under {suite2p_folder!r}")

    for ops_path in ops_files:
        folder = os.path.dirname(ops_path)
        print(f"[SpinesGUI_after] Patching {ops_path!r}…")
        ops = np.load(ops_path, allow_pickle=True).item()

        ops["ops_path"] = ops_path

        for key in ("reg_file", "reg_file_chan2"):
            if key in ops and ops[key]:
                old = ops[key]
                ops[key] = os.path.join(folder, os.path.basename(old))
                print(f"  • {key}: {old!r} → {ops[key]!r}")

        np.save(ops_path, ops)

    print("[SpinesGUI_after] All ops.npy files have been updated.")


def main():
    # --------------------------------------------------
    # 🔹 SET THESE MANUALLY (like before)
    # --------------------------------------------------
    userID='rubencorreia'
    expID='2026-02-24_01_ESRC027'

    combined = True              # True if this is a combined Suite2p run
    delete_combined = True      # True to delete suite2p_combined after verification
    # --------------------------------------------------

    move_original_suite2p_files(userID, expID)
    patch_all_ops_paths(userID, expID)

    if combined:
        # split + propagate SpinesGUI artifacts; deletion only if verification passes
        split_combined_suite2p_v3(
            userID,
            expID,
            copy_spinesgui_artifacts=True,
            delete_combined=bool(delete_combined),
        )
        print("🎉 All done! suite2p_combined has been split, and SpinesGUI artifacts were propagated.")
    else:
        print("🎉 All done! suite2p was patched and plane folders restored from SpinesGUI.")


if __name__ == "__main__":
    main()
