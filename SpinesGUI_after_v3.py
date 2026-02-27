import os
import glob
import shutil
import numpy as np
import organise_paths

# Keep the splitter as an independent script/module
from split_combined_s2p_with_checkpoints import split_combined_suite2p_v3


def move_original_suite2p_files(userID, expID):
    """
    1) Creates suite2p_original_files/ inside suite2p/
    2) Moves everything except SpinesGUI and suite2p_original_files into it
    3) Copies each plane-subfolder from SpinesGUI/ back into suite2p/
    """
    _, _, _, exp_dir_processed, _ = organise_paths.find_paths(userID, expID)
    suite2p_folder = os.path.join(exp_dir_processed, 'suite2p')
    new_folder = os.path.join(suite2p_folder, 'suite2p_original_files')
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created {new_folder!r}")

    for item in os.listdir(suite2p_folder):
        if item in ('SpinesGUI', 'suite2p_original_files'):
            continue
        src = os.path.join(suite2p_folder, item)
        print(f"Moving {src!r} → {new_folder!r}")
        shutil.move(src, new_folder)

    spines_gui = os.path.join(suite2p_folder, 'SpinesGUI')
    if os.path.isdir(spines_gui):
        for plane in os.listdir(spines_gui):
            src_plane = os.path.join(spines_gui, plane)
            if not os.path.isdir(src_plane):
                continue
            dst_plane = os.path.join(suite2p_folder, plane)
            os.makedirs(dst_plane, exist_ok=True)
            print(f"Copying contents of {src_plane!r} → {dst_plane!r}")
            for fn in os.listdir(src_plane):
                shutil.copy(os.path.join(src_plane, fn), dst_plane)

    print("All plane folders have been moved/copied into suite2p/.")  


def patch_all_ops_paths(userID, expID):
    """
    Recursively find every ops.npy under suite2p/ and:
      - reset ops['ops_path'] to its own file path
      - rebase reg_file and reg_file_chan2 if they exist
      - save back out
    """
    _, _, _, exp_dir_processed, _ = organise_paths.find_paths(userID, expID)
    suite2p_folder = os.path.join(exp_dir_processed, 'suite2p')

    pattern = os.path.join(suite2p_folder, '**', 'ops.npy')
    ops_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(ops_files)} ops.npy files under {suite2p_folder!r}")

    for ops_path in ops_files:
        folder = os.path.dirname(ops_path)
        print(f"Patching {ops_path!r}…")
        ops = np.load(ops_path, allow_pickle=True).item()

        ops['ops_path'] = ops_path

        for key in ('reg_file', 'reg_file_chan2'):
            if key in ops:
                old = ops[key]
                new = os.path.join(folder, os.path.basename(old))
                ops[key] = new
                print(f"  • {key}: {old!r} → {new!r}")

        np.save(ops_path, ops)
        print("  ✔ saved patched ops.npy")

    print("All ops.npy files have been updated.")


def def_main(
    userID='rubencorreia',
    expID='2025-11-11_01_ESRC022',
    *,
    combined=True,
    copy_spinesgui_artifacts=True,
    delete_suite2p_combined=False,
    verify_before_delete=True,
):
    """
    Full pipeline:
      1) Move original suite2p files aside, keep SpinesGUI/plane* copied back into suite2p/
      2) Patch ops.npy paths under suite2p/
      3) If combined=True, split suite2p_combined into per-expID suite2p/ folders
         + copy combined SpinesGUI artifacts into each expID (optional)
         + delete suite2p_combined only after checkpoints pass (optional)
    """
    move_original_suite2p_files(userID, expID)
    patch_all_ops_paths(userID, expID)

    if combined:
        split_combined_suite2p_v3(
            userID,
            expID,
            copy_spinesgui_artifacts=copy_spinesgui_artifacts,
            delete_suite2p_combined=delete_suite2p_combined,
            verify_before_delete=verify_before_delete,
        )
        print("🎉 All done! Split completed and ops.npy patched. You can now open Suite2p GUI per-expID.")
    else:
        print("🎉 All done! ops.npy patched. (combined=False so no splitting was performed.)")


if __name__ == '__main__':
    def_main()
