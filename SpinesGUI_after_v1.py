#!/usr/bin/env python3
"""Restore extracted ROI files into the Suite2p root folder."""

from __future__ import annotations

import argparse
import os
import shutil


def run_spinesgui_after(suite2p_folder: str) -> None:
    """Move original Suite2p files aside and restore extracted plane files."""
    suite2p_folder = os.path.abspath(os.path.expanduser(suite2p_folder))
    if not os.path.isdir(suite2p_folder):
        raise FileNotFoundError(f"Suite2p folder not found: {suite2p_folder}")

    spines_gui_subfolder = os.path.join(suite2p_folder, "SpinesGUI")
    if not os.path.isdir(spines_gui_subfolder):
        raise FileNotFoundError(f"SpinesGUI folder not found: {spines_gui_subfolder}")

    new_folder = os.path.join(suite2p_folder, "suite2p_original_files")
    os.makedirs(new_folder, exist_ok=True)
    print(f"[SpinesGUI_after] Created {new_folder}")

    for item in os.listdir(suite2p_folder):
        if item in ("SpinesGUI", "suite2p_original_files"):
            continue
        src = os.path.join(suite2p_folder, item)
        print(f"[SpinesGUI_after] Moving {src} -> {new_folder}")
        shutil.move(src, new_folder)

    for item in os.listdir(spines_gui_subfolder):
        src_plane = os.path.join(spines_gui_subfolder, item)
        if not os.path.isdir(src_plane):
            continue
        destination_folder = os.path.join(suite2p_folder, item)
        os.makedirs(destination_folder, exist_ok=True)
        print(f"[SpinesGUI_after] Restoring {src_plane} -> {destination_folder}")
        for filename in os.listdir(src_plane):
            shutil.copy2(
                os.path.join(src_plane, filename),
                os.path.join(destination_folder, filename),
            )

    print("[SpinesGUI_after] All files have been copied or moved to their correct directories")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-folder", help="Suite2p root folder")
    args = parser.parse_args()

    if args.root_folder:
        run_spinesgui_after(args.root_folder)
        return

    # Retain the previous no-argument manual workflow.
    from preprocess_pipeline.shared import paths

    user_id = "rubencorreia"
    exp_id = "2026-10-22_01_ESRC039"
    _, _, _, exp_dir_processed, _ = paths.find_paths(user_id, exp_id)
    run_spinesgui_after(os.path.join(exp_dir_processed, "suite2p"))


if __name__ == "__main__":
    main()
