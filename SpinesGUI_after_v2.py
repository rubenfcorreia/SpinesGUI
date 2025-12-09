import os
import glob
import shutil
import numpy as np
import organise_paths
from split_combined_s2p_modified import split_combined_suite2p_v2

def move_original_suite2p_files(userID, expID):
    """
    1. Creates suite2p_original_files/ inside suite2p/
    2. Moves everything except SpinesGUI and suite2p_original_files into it
    3. Copies each plane-subfolder from SpinesGUI/ back into suite2p/
    """
    _, _, _, exp_dir_processed, _ = organise_paths.find_paths(userID, expID)
    suite2p_folder = os.path.join(exp_dir_processed, 'suite2p')
    new_folder = os.path.join(suite2p_folder, 'suite2p_original_files')
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created {new_folder!r}")

    # Move all top-level items except SpinesGUI and our new folder
    for item in os.listdir(suite2p_folder):
        if item in ('SpinesGUI', 'suite2p_original_files'):
            continue
        src = os.path.join(suite2p_folder, item)
        print(f"Moving {src!r} → {new_folder!r}")
        shutil.move(src, new_folder)

    # Copy each plane folder out of SpinesGUI back into suite2p/
    spines_gui = os.path.join(suite2p_folder, 'SpinesGUI')
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

    # find every ops.npy
    pattern = os.path.join(suite2p_folder, '**', 'ops.npy')
    ops_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(ops_files)} ops.npy files under {suite2p_folder!r}")

    for ops_path in ops_files:
        folder = os.path.dirname(ops_path)
        print(f"Patching {ops_path!r}…")
        ops = np.load(ops_path, allow_pickle=True).item()

        # reset the ops_path
        ops['ops_path'] = ops_path

        # rebase any registration file paths
        for key in ('reg_file', 'reg_file_chan2'):
            if key in ops:
                old = ops[key]
                new = os.path.join(folder, os.path.basename(old))
                ops[key] = new
                print(f"  • {key}: {old!r} → {new!r}")

        np.save(ops_path, ops)
        print(f"  ✔ saved patched ops.npy")

    print("All ops.npy files have been updated.")

def main():
    userID = 'rubencorreia'
    expID  = '2025-11-11_01_ESRC022'  # or your combined experiment ID
    combined = True
    move_original_suite2p_files(userID, expID)
    patch_all_ops_paths(userID, expID)
    if combined:
        split_combined_suite2p_v2(userID, expID)
        print("🎉 All done! You can now run Suite2p GUI with every ops.npy correctly patched and every expID separated.")
    else:
        print("🎉 All done! You can now run Suite2p GUI with every ops.npy correctly patched")

if __name__ == '__main__':
    main()
